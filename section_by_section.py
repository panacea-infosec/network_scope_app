# scope_app_endpoints.py

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
    third_party_prompt, oof_sys_prompt, data_flow_prompt, risk_asmt_prompt
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
logger = logging.getLogger("fastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PCI DSS Audit Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

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

class ChatRequest(BaseModel):
    content_hash: str
    message: str
    document_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    cached_input_tokens: int

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
    scope_document: dict
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

def create_cached_prompt(system_prompt: str, static_context: str, dynamic_context: str, section_name: str, section_prompt: str) -> str:
    """
    Create a well-structured prompt for optimal caching.
    Static content (system prompt + file context) goes first for caching.
    Dynamic content (previously generated sections) goes last.
    """
    return f"""
{system_prompt}

EXTRACTED DOCUMENT INFORMATION:
{static_context}

PREVIOUSLY GENERATED SECTIONS:
{dynamic_context}

CURRENT SECTION TO GENERATE: {section_name}

INSTRUCTIONS:
{section_prompt}
"""

# def call_responses_api(
#     input_text: str,
#     model: str="gpt-4o",
#     **kwargs
# ) -> Tuple[str,int]:
#     """Returns (completion_text, cached_input_tokens)"""
#     resp = client.responses.create(
#         model=model,
#         input=input_text,
#         store=True,
#         **kwargs
#     )
#     # Get the output text
#     out_msg = resp.output_text
#     cached = resp.usage.input_tokens_details.cached_tokens
#     print(f"Input tokens: {resp.usage.input_tokens}")
#     print(f"Cached input tokens: {cached}")
#     return out_msg, cached
def call_responses_api(
    input_text: str,
    model: str="gpt-4o",
    **kwargs
) -> Tuple[str,int]:
    """Returns (completion_text, cached_input_tokens)"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.responses.create(
                model=model,
                input=input_text,
                store=True,
                **kwargs
            )
            out_msg = resp.output_text
            cached = resp.usage.input_tokens_details.cached_tokens
            print(f"Input tokens: {resp.usage.input_tokens}")
            print(f"Cached input tokens: {cached}")
            print(f"Output tokens: {resp.usage.output_tokens}")

            global total_input_tokens
            global total_cached_tokens
            global total_output_tokens
            total_input_tokens += resp.usage.input_tokens
            total_cached_tokens += resp.usage.input_tokens_details.cached_tokens
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
    gemini_parts = [Part.from_text(text=input_text)]
    content = Content(role="user", parts=gemini_parts)
    response = client_gemini.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=[content],
    )
    output_text = ""
    if response.candidates and response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
        parts = response.candidates[0].content.parts
        if parts and len(parts) > 0:
            first_part = parts[0]
            output_text = getattr(first_part, 'text', "") or ""
    # Gemini does not provide cached tokens, so return 0
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
        docs = [ {"id":k, **v} for k,v in global_state[ch]['document_data'].items() ]
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
    global_state[ch] = content_data

    docs = [{"id":k, "original_name":v["original_name"], "type":v["type"]} 
            for k,v in content_data['document_data'].items()]
    return UploadResponse(content_hash=ch, message="Files processed", documents=docs)


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    content = get_processed_content(req.content_hash)
    
    # Build the prompt based on whether we're analyzing a specific document or all documents
    if req.document_id:
        doc = content['document_data'].get(req.document_id)
        if not doc: 
            raise HTTPException(404, "Doc not found")
        
        system_content = f"You are an expert PCI DSS auditor. Analyze document {req.document_id}:\n{doc['text']}"
        history = content['doc_chat_histories'].setdefault(req.document_id, [])
    else:
        system_content = "You are an expert PCI DSS auditor. Use all uploaded documents to answer."
        history = content['chat_history']

    # Build the complete prompt
    prompt_parts = [system_content]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        prompt_parts.append(f"User: {user_msg}")
        prompt_parts.append(f"Assistant: {assistant_msg}")
    
    # Add current user message
    prompt_parts.append(f"User: {req.message}")
    prompt_parts.append("Assistant:")
    
    complete_prompt = "\n\n".join(prompt_parts)

    out, cached = call_responses_api(complete_prompt)
    
    # Store the conversation
    if req.document_id:
        content['doc_chat_histories'][req.document_id].append((req.message, out))
    else:
        content['chat_history'].append((req.message, out))

    return ChatResponse(response=out, cached_input_tokens=cached)


@app.post("/generate_full_report", response_model=FullReportResponse)
async def generate_full_report(req: FullReportRequest):
    content = get_processed_content(req.content_hash)
    
    # Define the order of sections for incremental generation
    section_order = [
        "Objective",
        "Business Overview", 
        "Cardholder Data Environment",
        "Connected Systems",
        "Third-Party Services",
        "Out-of-Scope Systems",
        "Data Flows",
        "Risk Assessment Summary",
       
    ]
    
    # Map section names to their prompts
    prompts = {
        "Objective": objective_prompt,
        "Business Overview": business_overview_prompt,
        "Cardholder Data Environment": cde_prompt,
        "Connected Systems": connect_sys_prompt,
        "Third-Party Services": third_party_prompt,
        "Out-of-Scope Systems": oof_sys_prompt,
        "Data Flows": data_flow_prompt,
        "Risk Assessment Summary": risk_asmt_prompt,
       
    }
    
    # Get processed data (text and images)
    text, images = content['processed_data']
    
    # Build the static prefix ONCE (system prompt + extracted text + images)
    static_prefix = f"{system_prompt}\n\nEXTRACTED DOCUMENT INFORMATION:\n{text}\n"
    if images:
        static_prefix += "\nIMAGES:\n"
        for i, (mime_type, _) in enumerate(images):
            static_prefix += f"\n[Image {i+1}: {mime_type} included and available to the auditor]\n"

    
    total_cached = 0
    generated_sections = {}
    
    # Generate sections incrementally
    for section_name in section_order:
        if section_name not in content.get('reports', {}):
            # Build dynamic context (previously generated sections)
            # dynamic_context = ""
            # if generated_sections:
            #     for prev_section, prev_content in generated_sections.items():
            #         dynamic_context += f"\n### {prev_section}\n{prev_content}\n"
            
            # Section-specific instructions
            section_instructions = (
                f"\nCURRENT SECTION TO GENERATE: {section_name}\n\n"
                f"INSTRUCTIONS:\n{prompts[section_name]}\n"
            )
            
            # Final prompt: static prefix + dynamic context + section instructions
            # complete_prompt = static_prefix + dynamic_context + section_instructions
            complete_prompt = static_prefix + section_instructions
            
            out, cached = call_responses_api(
                input_text=complete_prompt,
                model="gpt-4o"
            )
            logger.info(f"Generated {section_name}: cached_tokens={cached}")
            generated_sections[section_name] = out
            content.setdefault('reports', {})[section_name] = out
            total_cached += cached
        else:
            generated_sections[section_name] = content['reports'][section_name]
    
    # Create the final scope document
    scope_doc = {
        "id": str(uuid.uuid4()),
        "project_name": req.project_name,
        "qsa_name": req.qsa_name,
        "date": req.date,
        **generated_sections
    }
    
    logger.info(f"Full report generated: total_cached_tokens={total_cached}")
    
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
        scope_document=scope_doc,
        cached_input_tokens=total_cached,
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