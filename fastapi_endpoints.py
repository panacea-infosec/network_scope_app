from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
from openai import OpenAI
import pandas as pd
import fitz
from PyPDF2 import PdfReader
from PIL import Image
import openpyxl
import uuid
import tempfile
import hashlib
import time
from typing import Dict, List, Tuple, Optional
import re
import json
from dotenv import load_dotenv

load_dotenv()

# Import scope prompts
from scoping_prompts import (
    objective_prompt, business_overview_prompt, cde_prompt, connect_sys_prompt,
    third_party_prompt, oof_sys_prompt, data_flow_prompt, risk_asmt_prompt,
    asmp_exc_prompt, comp_val_prompt, roles_prompt, nextstep_prompt
)

app = FastAPI(title="PCI DSS Audit Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Global state management without sessions
# Key: content_hash, Value: ProcessedContent
global_state: Dict[str, dict] = {}

# Cleanup interval (1 hour)
CLEANUP_INTERVAL = 3600

# Data structures
class ProcessedContent:
    def __init__(self):
        self.content_hash: str = ""
        self.processed_data: Tuple[str, List[Tuple[str, str]]] = ("", [])
        self.document_data: Dict[str, Dict] = {}
        self.chat_history: List[Tuple[str, str]] = []
        self.doc_chat_histories: Dict[str, List[Tuple[str, str]]] = {}
        self.reports: Dict[str, str] = {}
        self.last_access: float = time.time()
        self.temp_dir: str = ""

class UploadResponse(BaseModel):
    content_hash: str
    message: str
    documents: List[Dict]

class ChatRequest(BaseModel):
    content_hash: str
    message: str
    document_id: Optional[str] = None

class ReportSectionRequest(BaseModel):
    content_hash: str
    section: str

class FullReportRequest(BaseModel):
    content_hash: str
    project_name: str
    qsa_name: str
    date: str

class FullReportResponse(BaseModel):
    id: str
    project_name: str
    qsa_name: str
    date: str
    scope_document: dict

# Utility functions
def generate_content_hash(file_contents: List[bytes]) -> str:
    """Generate a hash based on file contents to use as identifier"""
    hasher = hashlib.sha256()
    for content in file_contents:
        hasher.update(content)
    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity

def cleanup_old_data():
    """Remove old data to prevent memory bloat"""
    current_time = time.time()
    to_remove = []
    for content_hash, data in global_state.items():
        if current_time - data['last_access'] > CLEANUP_INTERVAL:
            # Clean up temp directory
            if data['temp_dir'] and os.path.exists(data['temp_dir']):
                import shutil
                shutil.rmtree(data['temp_dir'], ignore_errors=True)
            to_remove.append(content_hash)
    
    for content_hash in to_remove:
        del global_state[content_hash]

def get_processed_content(content_hash: str) -> dict:
    """Get processed content by hash, raise exception if not found"""
    cleanup_old_data()  # Clean up old data on each access
    
    if content_hash not in global_state:
        raise HTTPException(status_code=404, detail="Content not found. Please upload files first.")
    
    content = global_state[content_hash]
    content['last_access'] = time.time()
    return content

# File processing functions (same as original)
def process_image(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
    with open(file_path, "rb") as f:
        return (mime_type, base64.b64encode(f.read()).decode('utf-8'))

def process_excel(file_path: str) -> str:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        return "\n".join([f"Sheet: {name}\n{xls.parse(name).to_string()}" for name in xls.sheet_names])
    except Exception as e:
        return f"Error processing Excel: {e}"

def process_pdf(file_path: str) -> Tuple[str, List[str]]:
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        extracted_images = []
        for page in doc:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.width > 300 and pix.height > 300:
                    image_filename = f"temp_img_{xref}.png"
                    pix.save(image_filename)
                    extracted_images.append(image_filename)
                pix = None
        return full_text, extracted_images
    except Exception as e:
        return f"Error processing PDF: {e}", []

def process_files(file_paths: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    text, images = "", []
    for path in file_paths:
        base = os.path.basename(path)
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(process_image(path))
        elif path.lower().endswith(('.xlsx', '.xls')):
            text += f"\n--- Excel File: {base} ---\n"
            text += process_excel(path) + "\n"
        elif path.lower().endswith('.pdf'):
            text += f"\n--- PDF File: {base} ---\n"
            pdf_text, pdf_images = process_pdf(path)
            text += pdf_text + "\n"
            for img in pdf_images:
                file_paths.append(img)
                images.append(process_image(img))
    return text, images

def analyze_with_gpt4o(prompt: str, text: str, images: List[Tuple[str, str]], history=None) -> str:
    content = [{"type": "text", "text": f"Context:\n{text}\n\nQuestion: {prompt}"}]
    for mime, b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": content})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.3
    )
    return response.choices[0].message.content

# API Endpoints

@app.post("/upload_files", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files, return content hash for subsequent requests"""
    
    # Read file contents to generate hash
    file_contents = []
    file_data = []
    
    for file in files:
        content = await file.read()
        file_contents.append(content)
        file_data.append((file.filename, content))
        # Reset file pointer for later use
        await file.seek(0)
    
    # Generate content hash
    content_hash = generate_content_hash(file_contents)
    
    # Check if we already processed this content
    if content_hash in global_state:
        existing_data = global_state[content_hash]
        existing_data['last_access'] = time.time()
        return UploadResponse(
            content_hash=content_hash,
            message="Files already processed",
            documents=[{
                "id": name,
                "original_name": data['original_name'],
                "type": data.get('type', 'FILE'),
                "parent": data.get('parent_pdf')
            } for name, data in existing_data['document_data'].items()]
        )
    
    # Create temporary directory for this content
    temp_dir = os.path.join(tempfile.gettempdir(), f"pci_dss_{content_hash}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize content data
    content_data = {
        'content_hash': content_hash,
        'processed_data': ("", []),
        'document_data': {},
        'chat_history': [],
        'doc_chat_histories': {},
        'reports': {},
        'last_access': time.time(),
        'temp_dir': temp_dir
    }
    
    processed_files = []
    
    # Process each file
    for file, (filename, file_content) in zip(files, file_data):
        try:
            original_name = os.path.basename(filename)
            clean_name = re.sub(r'[^\w\.-]', '_', original_name)
            
            # Handle duplicate names
            base, ext = os.path.splitext(clean_name)
            counter = 0
            final_name = clean_name
            while final_name in content_data['document_data']:
                counter += 1
                final_name = f"{base}_{counter}{ext}"
            
            # Save file to temp directory
            file_path = os.path.join(temp_dir, final_name)
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Process file content according to file type
            text, images = "", []
            if final_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(process_image(file_path))
                content_data['document_data'][final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": '',
                    "images": images,
                    "type": "IMAGE"
                }
            elif final_name.lower().endswith(('.xlsx', '.xls')):
                text = process_excel(file_path)
                content_data['document_data'][final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": text,
                    "images": [],
                    "type": "EXCELSHEET"
                }
            elif final_name.lower().endswith('.pdf'):
                text, pdf_images = process_pdf(file_path)
                # Process extracted images
                for i, img_path in enumerate(pdf_images):
                    try:
                        pdf_base = os.path.splitext(final_name)[0]
                        img_original = f"Image {i+1} from {original_name}"
                        img_clean = f"{pdf_base}_image_{i+1}.png"
                        
                        # Handle duplicates
                        counter = 0
                        while img_clean in content_data['document_data']:
                            counter += 1
                            img_clean = f"{pdf_base}_image_{i+1}_{counter}.png"
                        
                        # Move image to temp directory
                        new_img_path = os.path.join(temp_dir, img_clean)
                        os.rename(img_path, new_img_path)
                        
                        # Process and store image metadata
                        images = [process_image(new_img_path)]
                        content_data['document_data'][img_clean] = {
                            "original_name": img_original,
                            "path": new_img_path,
                            "text": "",
                            "images": images,
                            "parent_pdf": final_name,
                            "type": "IMAGE"
                        }
                        processed_files.append(img_clean)
                    except Exception as e:
                        print(f"Error processing extracted image: {str(e)}")

                content_data['document_data'][final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": text,
                    "images": [],
                    "type": "PDF"
                }
            processed_files.append(final_name)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing {file.filename}: {str(e)}"
            )

    # Process combined files data
    all_paths = [v['path'] for v in content_data['document_data'].values()]
    combined_text, combined_images = process_files(all_paths)
    content_data['processed_data'] = (combined_text, combined_images)
    
    # Store in global state
    global_state[content_hash] = content_data
    
    return UploadResponse(
        content_hash=content_hash,
        message="Files processed successfully",
        documents=[{
            "id": name,
            "original_name": data['original_name'],
            "type": data.get('type', 'FILE'),
            "parent": data.get('parent_pdf')
        } for name, data in content_data['document_data'].items()]
    )

@app.post("/generate_report_section")
async def generate_report_section(request: ReportSectionRequest):
    """Generate a specific report section"""
    try:
        content = get_processed_content(request.content_hash)
    except HTTPException as e:
        raise e
    
    # Map scope section names to prompts
    section_prompts = {
        "Objective": objective_prompt,
        "Business Overview": business_overview_prompt,
        "Cardholder Data Environment": cde_prompt,
        "Connected Systems": connect_sys_prompt,
        "Third Parties": third_party_prompt,
        "Out-of-Scope Systems": oof_sys_prompt,
        "Data Flows": data_flow_prompt,
        "Risk Assessment": risk_asmt_prompt,
        "Assumptions/Exclusions": asmp_exc_prompt,
        "Compliance Validation": comp_val_prompt,
        "Stakeholders": roles_prompt,
        "Next Steps": nextstep_prompt
    }
    
    if request.section not in section_prompts:
        raise HTTPException(status_code=400, detail="Invalid section name")
    
    # Check if report already exists in cache
    if request.section in content['reports']:
        return {"section": request.section, "content": content['reports'][request.section]}
    
    try:
        text, images = content['processed_data']
        response = analyze_with_gpt4o(section_prompts[request.section], text, images)
        content['reports'][request.section] = response
        return {"section": request.section, "content": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating section: {str(e)}")

@app.post("/generate_full_report", response_model=FullReportResponse)
async def generate_full_report(request: FullReportRequest):
    """Generate complete report with all sections"""
    try:
        content = get_processed_content(request.content_hash)
    except HTTPException as e:
        raise e
    
    # Define all section prompts
    section_prompts = {
        "Objective": objective_prompt,
        "Business Overview": business_overview_prompt,
        "Cardholder Data Environment": cde_prompt,
        "Connected Systems": connect_sys_prompt,
        "Third Parties": third_party_prompt,
        "Out-of-Scope Systems": oof_sys_prompt,
        "Data Flows": data_flow_prompt,
        "Risk Assessment": risk_asmt_prompt,
        "Assumptions/Exclusions": asmp_exc_prompt,
        "Compliance Validation": comp_val_prompt,
        "Stakeholders": roles_prompt,
        "Next Steps": nextstep_prompt
    }

    # Generate all sections
    try:
        text, images = content['processed_data']
        for section, prompt in section_prompts.items():
            if section not in content['reports']:
                response = analyze_with_gpt4o(prompt, text, images)
                content['reports'][section] = response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

    return FullReportResponse(
        id=str(uuid.uuid4()),
        project_name=request.project_name,
        qsa_name=request.qsa_name,
        date=request.date,
        scope_document={
            "id": str(uuid.uuid4()),
            **content['reports']
        }
    )

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """Handle chat with GPT - general or document-specific"""
    try:
        content = get_processed_content(request.content_hash)
    except HTTPException as e:
        raise e
    
    try:
        if request.document_id:
            # Document-specific chat
            if request.document_id not in content['document_data']:
                raise HTTPException(status_code=400, detail="Document not found")
            
            doc_data = content['document_data'][request.document_id]
            text = doc_data['text']
            images = doc_data['images']
            
            # Get or create document-specific chat history
            if request.document_id not in content['doc_chat_histories']:
                content['doc_chat_histories'][request.document_id] = []
            
            history = content['doc_chat_histories'][request.document_id]
            
            messages = [{
                "role": "system",
                "content": f"""You are an expert PCI DSS auditor. 
                Analyze document: {request.document_id}
                Content: {text}
                Scan everything present in the file: figures, nodes, boxes, text, lines, etc.
                Note all components present in the network.
                Note all relationships between the components in the network.
                Do not add any information that is not present in the file originally.
                Only use information from this document to answer questions."""
            }]
            
            for user_msg, bot_resp in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_resp})
            
            response = analyze_with_gpt4o(request.message, text, images, messages)
            history.append((request.message, response))
            content['doc_chat_histories'][request.document_id] = history
        else:
            # General chat with all documents
            text, images = content['processed_data']
            messages = [{
                "role": "system",
                "content": """You are an expert PCI DSS auditor.
                The following information is extracted from several files. 
                Each file's content is preceded by a header indicating the file type and filename.
                Use these headers to distinguish between the documents and answer questions accordingly.
                Answer questions based on all uploaded documents.
                If user's question is irrelevant to the topic, 
                politely remind user to stick to asking questions about the given documents in a PCI DSS audit context."""
            }]
            
            for user_msg, bot_resp in content['chat_history']:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_resp})
            
            response = analyze_with_gpt4o(request.message, text, images, messages)
            content['chat_history'].append((request.message, response))
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/list_documents")
async def list_documents(content_hash: str):
    """List all processed documents for a given content hash"""
    try:
        content = get_processed_content(content_hash)
    except HTTPException as e:
        raise e
    
    return {
        "documents": [{
            "id": name,
            "original_name": data['original_name'],
            "type": data.get('type', 'FILE'),
            "parent": data.get('parent_pdf')
        } for name, data in content['document_data'].items()]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_contents": len(global_state)}

@app.delete("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old data"""
    initial_count = len(global_state)
    cleanup_old_data()
    cleaned_count = initial_count - len(global_state)
    return {"message": f"Cleaned up {cleaned_count} old content entries"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app) # , host="0.0.0.0", port=8000