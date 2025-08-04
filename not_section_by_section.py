from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, time, uuid, hashlib, tempfile, shutil, requests
from typing import List, Optional, Dict, Tuple
import base64, json
import logging
from google import genai
from google.genai.types import Content, Part
import pandas as pd
import fitz
from PyPDF2 import PdfReader
from PIL import Image
import openpyxl
from openai import OpenAI
from dotenv import load_dotenv

# Load your PCI‐DSS prompts
from updated_prompts import (
    system_prompt, objective_prompt, business_overview_prompt, cde_prompt, connect_sys_prompt,
    third_party_prompt, oof_sys_prompt, data_flow_prompt, risk_asmt_prompt,
)

load_dotenv()

logger = logging.getLogger("fastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PCI DSS Audit Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# GEMINI API KEY
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenAI client (Responses API)
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Global token counters
total_input_tokens = 0
total_cached_tokens = 0
total_output_tokens = 0

# In‐memory store
global_state: Dict[str, dict] = {}
CLEANUP_INTERVAL = 3600  # 1h

# ------ Data models ------

class UploadRequest(BaseModel):
    s3_urls: List[str]

class UploadResponse(BaseModel):
    content_hash: str
    message: str
    documents: List[Dict]

class FullReportRequest(BaseModel):
    content_hash: str
    project_name: Optional[str] = None
    qsa_name: Optional[str] = None
    date: Optional[str] = None

class FullReportResponse(BaseModel):
    id: str
    project_name: Optional[str]
    qsa_name: Optional[str]
    date: Optional[str]
    scope_document: str
    cached_input_tokens: int
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None

# ------ Utilities ------

def cleanup_old_data():
    now = time.time()
    expired = [
        key for key, data in global_state.items()
        if now - data['last_access'] > CLEANUP_INTERVAL
    ]
    for key in expired:
        temp_dir = global_state[key].get('temp_dir')
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        del global_state[key]

def get_processed_content(content_hash: str) -> dict:
    cleanup_old_data()
    if content_hash not in global_state:
        raise HTTPException(status_code=404, detail="Content not found.")
    global_state[content_hash]['last_access'] = time.time()
    return global_state[content_hash]

def generate_content_hash(contents: List[bytes]) -> str:
    h = hashlib.sha256()
    for b in contents: h.update(b)
    return h.hexdigest()[:16]

# ----- File Download & Type Detection -----

EXT_REGEX = re.compile(r"\.(png|jpe?g|pdf|xlsx?|xls)", re.IGNORECASE)

def download_s3_url(url: str, dest_dir: str) -> Tuple[str,str]:
    """Download file, infer extension, save, return (local_path, ext)"""
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise HTTPException(400, f"Failed to download {url}")
    # Try regex first
    m = EXT_REGEX.search(url)
    ext = m.group(1).lower() if m else None
    if not ext:
        # fallback to content-type
        ct = r.headers.get("Content-Type","")
        if "pdf" in ct: ext="pdf"
        elif "spreadsheet" in ct or "excel" in ct: ext="xlsx"
        elif "image/jpeg" in ct: ext="jpeg"
        elif "image/png" in ct: ext="png"
        else: ext="bin"
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(dest_dir, fname)
    with open(path,"wb") as f:
        for chunk in r.iter_content(1024*32): f.write(chunk)
    return path, ext

# ----- Existing processing fns (unchanged but simplified) -----

def process_image(fp:str)->Tuple[str,str]:
    ext=os.path.splitext(fp)[1].lower()
    mime="image/jpeg" if ext in ('.jpg','.jpeg') else "image/png"
    data=base64.b64encode(open(fp,"rb").read()).decode()
    return mime, data

def process_excel(fp:str)->str:
    xls=pd.ExcelFile(fp,engine='openpyxl')
    out=[]
    for name in xls.sheet_names:
        df=xls.parse(name)
        out.append(f"Sheet: {name}\n{df.to_string()}")
    return "\n\n".join(out)

def process_pdf(fp:str)->Tuple[str,List[str]]:
    doc=fitz.open(fp)
    txt = "".join(page.get_text() for page in doc)
    imgs=[]
    for page in doc:
        for imginfo in page.get_images(full=True):
            xref=imginfo[0]
            pix=fitz.Pixmap(doc,xref)
            if pix.width>300 and pix.height>300:
                imgp=f"{fp}_{xref}.png"
                pix.save(imgp); imgs.append(imgp)
            pix=None
    return txt, imgs

def process_files(paths:List[str]) -> Tuple[str,List[Tuple[str,str]]]:
    text=""
    images=[]
    for p in list(paths):
        low=p.lower()
        if low.endswith(('.png','.jpg','.jpeg')):
            images.append(process_image(p))
        elif low.endswith(('.xls','.xlsx')):
            text+= f"\n\n--- EXCEL {os.path.basename(p)} ---\n"
            text+= process_excel(p)
        elif low.endswith('.pdf'):
            text+= f"\n\n--- PDF {os.path.basename(p)} ---\n"
            pdf_txt, pdf_imgs = process_pdf(p)
            text+= pdf_txt
            for ip in pdf_imgs:
                paths.append(ip)
    return text, images

# ----- GPT via Responses API & caching -----

def call_responses_api(
    messages: List[Dict],
    model: str="gpt-4o",
    **kwargs
) -> Tuple[str,int]:
    """Returns (completion_text, cached_tokens)"""
    global total_input_tokens, total_cached_tokens, total_output_tokens
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.responses.create(
                model=model,
                input=messages,
                store=True,
                **kwargs
            )
            out_msg = resp.output_text
            cached = resp.usage.input_tokens_details.cached_tokens
            print(f"Input tokens: {resp.usage.input_tokens}")
            print(f"Cached input tokens: {cached}")
            print(f"Output tokens: {resp.usage.output_tokens}")
            total_input_tokens += resp.usage.input_tokens
            total_cached_tokens += cached
            total_output_tokens += resp.usage.output_tokens
            return out_msg, cached
        except Exception as e:
            print(f"OpenAI failed: {e}. Trying Gemini fallback...")
    else:
        print("No OpenAI API key found. Using Gemini fallback...")
    if not GEMINI_API_KEY:
        raise Exception("No AI service available.")
    client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    # Gemini expects a Content object with parts
    flattened_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        for part in msg.get("content", []):
            if part["type"] == "input_text":
                flattened_text += f"\n[{role.upper()} TEXT]: {part['text']}"
            elif part["type"] == "input_image":
                flattened_text += f"\n[{role.upper()} IMAGE]: (base64 image omitted)"
            else:
                flattened_text += f"\n[{role.upper()} UNKNOWN PART]"
    gemini_parts = [Part.from_text(text=flattened_text)]
    content = Content(role="user", parts=gemini_parts)
    response = client_gemini.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[content],
    )
    output_text = ""
    if response.candidates and response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
        parts = response.candidates[0].content.parts
        if parts and len(parts) > 0:
            first_part = parts[0]
            output_text = getattr(first_part, 'text', "") or ""
    # Gemini does not provide cached tokens, so return 0
    total_output_tokens += response.usage.output_tokens
    return output_text, 0

# ------ API Endpoints ------

@app.post("/upload_files", response_model=UploadResponse)
async def upload_files(req: UploadRequest):
    # download all into a fresh temp dir
    temp_dir = os.path.join(tempfile.gettempdir(), "pci_"+uuid.uuid4().hex)
    os.makedirs(temp_dir, exist_ok=True)
    local_paths, raw_bytes = [], []
    for url in req.s3_urls:
        path, ext = download_s3_url(url, temp_dir)
        local_paths.append(path)
        raw_bytes.append(open(path,"rb").read())

    ch = generate_content_hash(raw_bytes)
    if ch in global_state:
        global_state[ch]['last_access']=time.time()
        # docs = [ {"id":k, **v} for k,v in global_state[ch]['document_data'].items() ]
        docs = [
           {"id": k, "original_name": v["original_name"], "type": v["type"]}
           for k, v in global_state[ch]['document_data'].items()
        ]
        return UploadResponse(content_hash=ch, message="Already processed", documents=docs)

    # process each file & fill document_data
    content_data = dict(
        content_hash=ch, document_data={}, chat_history=[], doc_chat_histories={},
        reports={}, last_access=time.time(), temp_dir=temp_dir
    )
    for lp in local_paths:
        nm=os.path.basename(lp)
        low=nm.lower()
        text, images = "", []
        if low.endswith(('.png','.jpg','.jpeg')):
            mime, b64 = process_image(lp)
            images=[(mime,b64)]
            ttype="IMAGE"
        elif low.endswith(('.xls','.xlsx')):
            text=process_excel(lp); ttype="EXCELSHEET"
        elif low.endswith('.pdf'):
            text, pdf_imgs = process_pdf(lp)
            # embed child images
            for pi in pdf_imgs:
                m,b=process_image(pi)
                images.append((m,b))
            ttype="PDF"
        else:
            text=f"(Unsupported file type: {nm})"; ttype="UNKNOWN"

        content_data['document_data'][nm]=dict(
            original_name=nm, path=lp, text=text, images=images, type=ttype
        )

    # combine all
    allp=[v['path'] for v in content_data['document_data'].values()]
    comb_txt, comb_imgs = process_files(allp)
    content_data['processed_data'] = (comb_txt, comb_imgs)
    
    # Create and store the complete report prompt for caching
    prompts_dict = {
        "Objective": objective_prompt,
        "Business Overview": business_overview_prompt,
        "Cardholder Data Environment": cde_prompt,
        "Connected Systems": connect_sys_prompt,
        "Third-Party Services": third_party_prompt,
        "Out-of-Scope Systems": oof_sys_prompt,
        "Data Flows": data_flow_prompt,
        "Risk Assessment Summary": risk_asmt_prompt,
        
    }
    
    complete_prompt = f"""{system_prompt}

EXTRACTED DOCUMENT INFORMATION:
{comb_txt}"""
    
    # Add images if present
    if comb_imgs:
        complete_prompt += "\n\nIMAGES:"
        for i, (mime_type, base64_data) in enumerate(comb_imgs):
            complete_prompt += f"\n\nImage {i+1} ({mime_type}):\n{base64_data}"
    
    complete_prompt += """

TASK: Generate a complete PCI DSS scope document report by analyzing all the provided information above. 

INSTRUCTIONS:
1. Carefully analyze all the extracted document information and images provided
2. Generate the complete scope document report in a single response
3. Structure your response with clear sections using markdown formatting
4. Follow the section-wise instructions provided below for each section
5. Ensure each section is comprehensive and based on the analyzed information
6. If information is missing for any section, clearly state what additional information is required

GENERATE THE COMPLETE SCOPE DOCUMENT WITH THE FOLLOWING SECTIONS:

## 1. Objective
"""
    complete_prompt += objective_prompt + "\n"
    
    complete_prompt += "\n## 2. Business Overview\n"
    complete_prompt += business_overview_prompt + "\n"
    
    complete_prompt += "\n## 3. Cardholder Data Environment\n"
    complete_prompt += cde_prompt + "\n"
    
    complete_prompt += "\n## 4. Connected Systems\n"
    complete_prompt += connect_sys_prompt + "\n"
    
    complete_prompt += "\n## 5. Third-Party Services\n"
    complete_prompt += third_party_prompt + "\n"
    
    complete_prompt += "\n## 6. Out-of-Scope Systems\n"
    complete_prompt += oof_sys_prompt + "\n"
    
    complete_prompt += "\n## 7. Data Flows\n"
    complete_prompt += data_flow_prompt + "\n"
    
    complete_prompt += "\n## 8. Risk Assessment Summary\n"
    complete_prompt += risk_asmt_prompt + "\n"
    
    
    
    complete_prompt += """

IMPORTANT FORMATTING REQUIREMENTS:
- Use markdown formatting with appropriate headers (##, ###)
- Provide detailed analysis for each section based on the provided information
- Maintain professional tone throughout the document
- Include specific details from the analyzed documents and images
- If any required information is missing, clearly list it under each relevant section

BEGIN GENERATING THE COMPLETE SCOPE DOCUMENT NOW:
"""
    
    content_data['complete_report_prompt'] = complete_prompt
    
    global_state[ch] = content_data

    # docs = [{"id":k, "original_name":v["original_name"], "type":v["type"]} 
    #         for k,v in content_data['document_data'].items()]
    docs = [{"id":k, "original_name":v["original_name"], "type":v["type"]} 
        for k,v in content_data['document_data'].items()]
    return UploadResponse(content_hash=ch, message="Files processed", documents=docs)



@app.post("/generate_full_report", response_model=FullReportResponse)
async def generate_full_report(req: FullReportRequest):
    content = get_processed_content(req.content_hash)
    
    section_order = [\
        "Objective", "Business Overview", "Cardholder Data Environment", "Connected Systems",
        "Third-Party Services", "Out-of-Scope Systems", "Data Flows", "Risk Assessment Summary",
       
    ]
    prompts = {\
        "Objective": objective_prompt,
        "Business Overview": business_overview_prompt,
        "Cardholder Data Environment": cde_prompt, 
        "Connected Systems": connect_sys_prompt,
        "Third-Party Services": third_party_prompt, 
        "Out-of-Scope Systems": oof_sys_prompt,
        "Data Flows": data_flow_prompt, 
        "Risk Assessment Summary": risk_asmt_prompt,
        
    }

    # Check if report is already generated and cached
    if 'full_report' in content.get('reports', {}):
        logger.info("Using cached full report")
        cached_report = content['reports']['full_report']
        return FullReportResponse(
            id=str(uuid.uuid4()),
            project_name=req.project_name,
            qsa_name=req.qsa_name,
            date=req.date,
            scope_document=cached_report['content'],
            cached_input_tokens=cached_report['cached_tokens'],
            total_input_tokens=cached_report.get('total_input_tokens'),
            total_output_tokens=cached_report.get('total_output_tokens')
        )
    
    text, images = content['processed_data']

    # Build the messages list for the OpenAI API call
    messages = []

    # System prompt part
    system_content_parts = [
        {"type": "input_text", "text": system_prompt},
        {"type": "input_text", "text": "EXTRACTED DOCUMENT INFORMATION:\n" + text}
    ]
    messages.append({"role": "system", "content": system_content_parts})

    # User content part (combining images and the main request)
    user_content_parts = []
    if images:
        for i, (mime_type, base64_data) in enumerate(images):
            user_content_parts.append({"type": "input_image", "image_url":  f"data:{ mime_type};base64,{base64_data}"})

    main_request_text = """
TASK: Generate a complete PCI DSS scope document report by analyzing all the provided information above. 

INSTRUCTIONS:
1. Carefully analyze all the extracted document information and images provided
2. Generate the complete scope document report in a single response
3. Structure your response with clear sections using markdown formatting
4. Follow the section-wise instructions provided below for each section
5. Ensure each section is comprehensive and based on the analyzed information
6. If information is missing for any section, clearly state what additional information is required

GENERATE THE COMPLETE SCOPE DOCUMENT WITH THE FOLLOWING SECTIONS:
"""
    for section_name in section_order:
        main_request_text += f"\n\n## {section_name}\n"
        main_request_text += prompts[section_name]
    
    main_request_text += """

IMPORTANT FORMATTING REQUIREMENTS:
- Use markdown formatting with appropriate headers (##, ###)
- Provide detailed analysis for each section based on the provided information
- Maintain professional tone throughout the document
- Include specific details from the analyzed documents and images
- If any required information is missing, clearly list it under each relevant section

BEGIN GENERATING THE COMPLETE SCOPE DOCUMENT NOW:
"""
    user_content_parts.append({"type": "input_text", "text": main_request_text})
    
    messages.append({"role": "user", "content": user_content_parts})
    
    # Log prompt details for debugging
    # For multimodal input, direct token estimation is harder, rely on API's cached_tokens
    logger.info(f"Generating complete scope document report...")
    
    out, cached = call_responses_api(
        messages=messages,
        model="gpt-4o"
    )
    
    logger.info(f"Complete report generated: cached_tokens={cached}")
    
    # Cache the generated report
    content.setdefault('reports', {})['full_report'] = {
        'content': out,
        'cached_tokens': cached,
        'generated_at': time.time(),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens
    }

    # Calculate total cost
    input_cost = (total_input_tokens / 1_000_000) * 2.50
    cached_cost = (total_cached_tokens / 1_000_000) * 1.25
    output_cost = (total_output_tokens / 1_000_000) * 10.00
    total_cost = input_cost + cached_cost + output_cost

    print(f"Total Input Tokens for full report: {total_input_tokens}")
    print(f"Total Cached Tokens for full report: {total_cached_tokens}")
    print(f"Total Output Tokens for full report: {total_output_tokens}")
    print(f"Total Input Cost for full report: ${input_cost:.4f}")
    print(f"Total Cached Cost for full report: ${cached_cost:.4f}")
    print(f"Total Output Cost for full report: ${output_cost:.4f}")
    print(f"Total Cost for full report: ${total_cost:.4f}")

    return FullReportResponse(
        id=str(uuid.uuid4()),
        project_name=req.project_name,
        qsa_name=req.qsa_name,
        date=req.date,
        scope_document=out,
        cached_input_tokens=cached,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens
    )


@app.get("/list_documents")
async def list_documents(content_hash: str):
    content = get_processed_content(content_hash)
    return {"documents": [
        {"id":k, "original_name":v["original_name"], "type":v["type"]}
        for k,v in content['document_data'].items()
    ]}


@app.get("/health")
async def health_check():
    return {"status":"healthy","active_contents":len(global_state)}


@app.delete("/cleanup")
async def manual_cleanup():
    before = len(global_state)
    cleanup_old_data()
    after = len(global_state)
    return {"cleaned": before-after}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app)