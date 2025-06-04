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
from typing import Dict, List, Tuple
import time
import re

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
openai_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key="sk-")

# Session management
sessions: Dict[str, dict] = {}
 
# Session timeout (1 hour)
#NOTE: Change as required
SESSION_TIMEOUT = 3600

# Session class to store data on FastAPI server
class SessionData:
    def __init__(self):
        self.processed_data: Tuple[str, List[Tuple[str, str]]] = ("", [])
        self.document_data: Dict[str, Tuple[str, str, List]] = {}
        self.chat_history: List[Tuple[str, str]] = []
        self.doc_chat_histories: Dict[str, List[Tuple[str, str]]] = {}
        self.reports: Dict[str, str] = {}
        self.last_access: float = time.time()

# Utility functions
def get_session(session_id: str) -> SessionData:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if time.time() - session.last_access > SESSION_TIMEOUT:
        del sessions[session_id]
        raise HTTPException(status_code=410, detail="Session expired")
    session.last_access = time.time()
    return session

# File processing functions (same as in Gradio app implementation)

# Process images to convert into base64 encoding
def process_image(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
    with open(file_path, "rb") as f:
        return (mime_type, base64.b64encode(f.read()).decode('utf-8'))

# Process excel sheets to extract text
def process_excel(file_path: str) -> str:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        return "\n".join([f"Sheet: {name}\n{xls.parse(name).to_string()}" for name in xls.sheet_names])
    except Exception as e:
        return f"Error processing Excel: {e}"

# Process PDF file to extract text and images
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

# Process uploaded files according to file type
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

# Function to send files (text and images) to GPT (to generate scope and chat)
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
# creates a new session with unique id
# the session data is stored in memory within the FastAPI application's runtime
# The sessions dictionary stores all session data in memory.
# Each session is identified by a unique session_id (UUID).
@app.post("/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData()
    return {"session_id": session_id}

# Endpoint to upload files(Image, PDF, Excel) and process them (extract text and images)
# creates pointers to file paths, extract images with file 
@app.post("/upload_files")
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e

    # Create session-specific storage directory
    if not hasattr(session, 'session_dir'):
        session.session_dir = os.path.join(tempfile.gettempdir(), f"pci_dss_{session_id}")
        os.makedirs(session.session_dir, exist_ok=True)

    processed_files = []
    for file in files:
        try:
            # get original filename
            original_name = os.path.basename(file.filename)
            clean_name = re.sub(r'[^\w\.-]', '_', original_name)
            
            # Handle duplicate names
            base, ext = os.path.splitext(clean_name)
            counter = 0
            final_name = clean_name
            while final_name in session.document_data:
                counter += 1
                final_name = f"{base}_{counter}{ext}"
            
            # Save file with final name
            file_path = os.path.join(session.session_dir, final_name)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Process file content according to file type
            text, images = "", []
            if final_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(process_image(file_path))
                # Update document_data for the Image
                session.document_data[final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": '',
                    "images": images,
                    "type": "IMAGE"
                }
            elif final_name.lower().endswith(('.xlsx', '.xls')):
                text = process_excel(file_path)
                # Update document_data for the Excel
                session.document_data[final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": text,
                    "images": [],
                    "type": "EXCELSHEET"
                }
            # PDF-specific processing    
            elif final_name.lower().endswith('.pdf'):
                text, pdf_images = process_pdf(file_path)
                # Process extracted images
                for i, img_path in enumerate(pdf_images):
                    try:
                        # Generate image name based on PDF
                        pdf_base = os.path.splitext(final_name)[0]
                        img_original = f"Image {i+1} from {original_name}"
                        img_clean = f"{pdf_base}_image_{i+1}.png"
                        
                        # Handle duplicates
                        counter = 0
                        while img_clean in session.document_data:
                            counter += 1
                            img_clean = f"{pdf_base}_image_{i+1}_{counter}.png"
                        
                        # Move image to session directory
                        new_img_path = os.path.join(session.session_dir, img_clean)
                        os.rename(img_path, new_img_path)
                        
                        # Process and store image metadata
                        images = [process_image(new_img_path)]
                        session.document_data[img_clean] = {
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

                # Update document_data for the PDF
                session.document_data[final_name] = {
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
    all_paths = [v['path'] for v in session.document_data.values()]
    combined_text, combined_images = process_files(all_paths)
    session.processed_data = (combined_text, combined_images)

    return {
        "message": "Files processed successfully",
        "documents": [{
            "id": name,
            "original_name": data['original_name'],
            "type": data.get('type', 'FILE'),
            "parent": data.get('parent_pdf')
        } for name, data in session.document_data.items()]
    }

# End point to generate scope report section according to section title provided
@app.post("/generate_report_section")
async def generate_report_section(session_id: str, section: str):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
    
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
        #"Assumptions/Exclusions": asmp_exc_prompt,
        #"Compliance Validation": comp_val_prompt,
        #"Stakeholders": roles_prompt,
        #"Next Steps": nextstep_prompt
    }
    
    if section not in section_prompts:
        raise HTTPException(status_code=400, detail="Invalid section name")
    
    if section in session.reports:
        return {"section": section, "content": session.reports[section]}
    
    try:
        text, images = session.processed_data
        response = analyze_with_gpt4o(section_prompts[section], text, images)
        session.reports[section] = response
        return {"section": section, "content": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating section: {str(e)}")
    
# End point to generate entire scope document report
# Add new Pydantic model for response structure
class FullReportResponse(BaseModel):
    id: str
    project_name: str
    qsa_name: str
    date: str
    scope_document: dict

# Add new endpoint to generate full report
@app.post("/generate_full_report", response_model=FullReportResponse)
async def generate_full_report(
    session_id: str, 
    project_name: str,
    qsa_name: str,
    date:str
):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
    
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
        # "Assumptions/Exclusions": asmp_exc_prompt,
        # "Compliance Validation": comp_val_prompt,
        # "Stakeholders": roles_prompt,
        # "Next Steps": nextstep_prompt
    }

    # Generate all sections
    try:
        text, images = session.processed_data
        for section, prompt in section_prompts.items():
            if section not in session.reports:
                response = analyze_with_gpt4o(prompt, text, images)
                session.reports[section] = response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

    return {
        "id": str(uuid.uuid4()),
        "project_name": project_name,
        "qsa_name": qsa_name,
        "scope_document": {
            "id": str(uuid.uuid4()),
            **session.reports
        }
    }

# End point to handle chat with GPT
# if document_id is selected (from list of processed documents), chat about the specific document
# else, takes all documents in context when answering user query
@app.post("/chat")
async def handle_chat(session_id: str, message: str, document_id: str = None):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
    
    try:
        if document_id:
            # Document-specific chat
            if document_id not in session.document_data:
                raise HTTPException(status_code=400, detail="Document not found")
            
            doc_data = session.document_data[document_id]
            text = doc_data['text']
            images = doc_data['images']
            # maintains history for each document seperately
            history = session.doc_chat_histories.get(document_id, [])
            
            messages = [{
                "role": "system",
                "content": f"""You are an expert PCI DSS auditor. 
                Analyze document: {document_id}\nContent: {text}.
                Only use information from this document to answer questions."""
            }]
            
            for user_msg, bot_resp in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_resp})
            
            response = analyze_with_gpt4o(message, text, images, messages)
            history.append((message, response))
            session.doc_chat_histories[document_id] = history
        else:
            # General chat
            text, images = session.processed_data
            messages = [{
                "role": "system",
                "content": "You are an expert PCI DSS auditor. Answer questions based on all uploaded documents."
            }]
            
            for user_msg, bot_resp in session.chat_history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_resp})
            
            response = analyze_with_gpt4o(message, text, images, messages)
            session.chat_history.append((message, response))
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# End point returns the list of all processed documents in session
@app.get("/list_documents")
async def list_documents(session_id: str):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
    
    return {
        "documents": [{
            "id": name,
            "original_name": data['original_name'],
            "type": data.get('type', 'FILE'),
            "parent": data.get('parent_pdf')
        } for name, data in session.document_data.items()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)#, host="0.0.0.0", port=8000