import os
import time
import uuid
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# OpenAI and Gemini
import openai
# Gemini 2.5 Flash official SDK imports
from google import genai
from google.genai import types
# import google.generativeai as genai
# from google.generativeai.types import Content, Part
from google.genai.types import Content, Part
import requests  # For OpenAI REST fallback

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="PCI DSS Audit Assistant API (Responses API)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache (replace with Redis for production)
CACHE_TTL = 3600  # 1 hour
cache: Dict[str, dict] = {}

# Data models
class FileMeta(BaseModel):
    id: str
    s3_url: str
    type: str  # e.g. 'PDF', 'IMAGE', 'EXCELSHEET'
    extracted_text: Optional[str] = None
    original_name: Optional[str] = None

class UploadRequest(BaseModel):
    files: List[FileMeta]

class UploadResponse(BaseModel):
    content_hash: str
    message: str
    documents: List[FileMeta]

class ChatRequest(BaseModel):
    content_hash: str
    message: str
    document_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str
    response_id: Optional[str] = None

# Add section prompts (import or define here)
section_prompts = {
    "Objective": "Summarize the objective of the PCI DSS audit based on the provided documents.",
    "Business Overview": "Provide a business overview relevant to PCI DSS scope.",
    "Cardholder Data Environment": "Describe the Cardholder Data Environment (CDE) as per the documents.",
    "Connected Systems": "List and describe all systems connected to the CDE.",
    "Third Parties": "Identify and summarize the role of third parties in the PCI DSS scope.",
    "Out-of-Scope Systems": "List and describe all out-of-scope systems.",
    "Data Flows": "Summarize the data flows relevant to PCI DSS.",
    "Risk Assessment": "Summarize the risk assessment findings.",
    "Assumptions/Exclusions": "List any assumptions or exclusions noted in the scope.",
    "Compliance Validation": "Summarize the compliance validation approach.",
    "Stakeholders": "List the stakeholders involved in the PCI DSS audit.",
    "Next Steps": "List the next steps for the PCI DSS audit project."
}

class ReportSectionRequest(BaseModel):
    content_hash: str
    section: str

class FullReportRequest(BaseModel):
    content_hash: str
    project_name: Optional[str] = None
    qsa_name: Optional[str] = None
    date: Optional[str] = None

class FullReportResponse(BaseModel):
    id: str
    project_name: Optional[str] = None
    qsa_name: Optional[str] = None
    date: Optional[str] = None
    scope_document: dict

# Utility functions
def cleanup_cache():
    now = time.time()
    to_delete = [k for k, v in cache.items() if now - v['last_access'] > CACHE_TTL]
    for k in to_delete:
        del cache[k]

def get_cache(content_hash: str) -> dict:
    cleanup_cache()
    if content_hash not in cache:
        raise HTTPException(status_code=404, detail="Content not found. Please upload files first.")
    cache[content_hash]['last_access'] = time.time()
    return cache[content_hash]

def generate_content_hash(files: List[FileMeta]) -> str:
    # Use UUID for demo; in production, hash file URLs and text for deduplication
    return str(uuid.uuid4())[:16]

# OpenAI and Gemini setup
def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    # The client will pick up GOOGLE_API_KEY from env if set
    return genai.Client(api_key=GEMINI_API_KEY)

def call_openai_responses_api(openai_api_key, openai_kwargs):
    """Call OpenAI Responses API using requests if SDK does not support it."""
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=openai_kwargs)
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} {response.text}")
    return response.json()

# API Endpoints
@app.post("/upload_files", response_model=UploadResponse)
def upload_files(request: UploadRequest):
    """Register S3 URLs and extracted text for files. Returns a content_hash for chat."""
    # Generate a content hash (could hash URLs+text for deduplication)
    content_hash = generate_content_hash(request.files)
    # Store in cache
    cache[content_hash] = {
        'files': request.files,
        'chat_history': [],
        'last_response_id': None,
        'last_access': time.time(),
    }
    return UploadResponse(
        content_hash=content_hash,
        message="Files registered successfully.",
        documents=request.files
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat endpoint using OpenAI Responses API with S3 URLs and fallback to Gemini 2.5 Flash."""
    try:
        session = get_cache(request.content_hash)
        files = session['files']
        chat_history = session['chat_history']
        last_response_id = session.get('last_response_id')

        # Prepare OpenAI input
        user_content = []
        if request.document_id:
            doc = next((f for f in files if f.id == request.document_id), None)
            if not doc:
                raise HTTPException(status_code=400, detail="Document not found.")
            if doc.extracted_text:
                user_content.append({"type": "input_text", "text": doc.extracted_text})
            if doc.type == "IMAGE":
                user_content.append({"type": "input_image", "image_url": doc.s3_url})
            elif doc.type == "PDF":
                pass
            user_content.append({"type": "input_text", "text": request.message})
        else:
            for doc in files:
                if doc.extracted_text:
                    user_content.append({"type": "input_text", "text": f"[{doc.type}] {doc.original_name or doc.id}:\n{doc.extracted_text}"})
                if doc.type == "IMAGE":
                    user_content.append({"type": "input_image", "image_url": doc.s3_url})
            user_content.append({"type": "input_text", "text": request.message})

        openai_input = [
            {
                "role": "user",
                "content": user_content
            }
        ]
        openai_kwargs = {
            "model": "gpt-4o-mini", # choose OpenAI model: o3, gpt-4o, gpt-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano
            "input": openai_input,
        }
        if last_response_id:
            openai_kwargs["previous_response_id"] = last_response_id

        try:
            if not OPENAI_API_KEY:
                raise Exception("No OpenAI key")
            # Use REST API for Responses endpoint (SDK may not support it)
            response_json = call_openai_responses_api(OPENAI_API_KEY, openai_kwargs)
            output_text = response_json["output"][0]["content"][0]["text"]
            response_id = response_json["id"]
            session['last_response_id'] = response_id
            session['chat_history'].append((request.message, output_text))
            return ChatResponse(response=output_text, model_used="openai-gpt-4o", response_id=response_id)
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}. Falling back to Gemini 2.5 Flash.")
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=503, detail="No AI service available.")
            client = get_gemini_client()
            # Prepare Gemini 2.5 Flash input as per SDK
            gemini_parts = []
            for part in user_content:
                if part["type"] == "input_text":
                    gemini_parts.append(Part.from_text(text=part["text"]))
                elif part["type"] == "input_image":
                    gemini_parts.append(Part.from_uri(file_uri=part["image_url"], mime_type="image/png"))
            content = Content(role="user", parts=gemini_parts)
            response = client.models.generate_content(
                model="models/gemini-2.5-flash-latest",
                contents=[content],
            )
            output_text = ""
            if response.candidates and response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
                parts = response.candidates[0].content.parts
                if parts and len(parts) > 0:
                    first_part = parts[0]
                    output_text = getattr(first_part, 'text', "") or ""
            session['chat_history'].append((request.message, output_text))
            return ChatResponse(response=output_text, model_used="gemini-2.5-flash")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(cache)}

@app.delete("/cleanup")
def cleanup():
    before = len(cache)
    cleanup_cache()
    after = len(cache)
    return {"message": f"Cleaned up {before - after} expired sessions."}

# Example: List documents for a session
@app.get("/list_documents")
def list_documents(content_hash: str):
    session = get_cache(content_hash)
    return {"documents": [f.dict() for f in session['files']]}

@app.post("/generate_report_section")
def generate_report_section(request: ReportSectionRequest):
    """Generate a specific report section using OpenAI/Gemini."""
    try:
        session = get_cache(request.content_hash)
        if "reports" not in session:
            session["reports"] = {}
        if request.section not in section_prompts:
            raise HTTPException(status_code=400, detail="Invalid section name")
        # Check cache
        if request.section in session["reports"]:
            return {"section": request.section, "content": session["reports"][request.section]}
        # Use all processed text/images for the report
        files = session['files']
        text = "\n".join([f.extracted_text or "" for f in files if f.extracted_text])
        images = [f for f in files if f.type == "IMAGE"]
        prompt = section_prompts[request.section]
        # Use chat logic for OpenAI/Gemini
        user_content = []
        if text:
            user_content.append({"type": "input_text", "text": text})
        for img in images:
            user_content.append({"type": "input_image", "image_url": img.s3_url})
        user_content.append({"type": "input_text", "text": prompt})
        openai_input = [
            {"role": "user", "content": user_content}
        ]
        openai_kwargs = {"model": "gpt-4o", "input": openai_input}
        try:
            if not OPENAI_API_KEY:
                raise Exception("No OpenAI key")
            response_json = call_openai_responses_api(OPENAI_API_KEY, openai_kwargs)
            output_text = response_json["output"][0]["content"][0]["text"]
            session["reports"][request.section] = output_text
            return {"section": request.section, "content": output_text}
        except Exception as e:
            logger.warning(f"OpenAI failed for report section: {e}. Falling back to Gemini.")
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=503, detail="No AI service available.")
            client = get_gemini_client()
            gemini_parts = []
            if text:
                gemini_parts.append(Part.from_text(text=text))
            for img in images:
                gemini_parts.append(Part.from_uri(file_uri=img.s3_url, mime_type="image/png"))
            gemini_parts.append(Part.from_text(text=prompt))
            content = Content(role="user", parts=gemini_parts)
            response = client.models.generate_content(
                model="models/gemini-2.5-flash-latest",
                contents=[content],
            )
            output_text = ""
            if response.candidates and response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
                parts = response.candidates[0].content.parts
                if parts and len(parts) > 0:
                    first_part = parts[0]
                    output_text = getattr(first_part, 'text', "") or ""
            session["reports"][request.section] = output_text
            return {"section": request.section, "content": output_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report section error: {e}")
        raise HTTPException(status_code=500, detail=f"Report section error: {e}")

@app.post("/generate_full_report", response_model=FullReportResponse)
def generate_full_report(request: FullReportRequest):
    """Generate complete report with all sections using OpenAI/Gemini."""
    import uuid
    try:
        session = get_cache(request.content_hash)
        if "reports" not in session:
            session["reports"] = {}
        files = session['files']
        text = "\n".join([f.extracted_text or "" for f in files if f.extracted_text])
        images = [f for f in files if f.type == "IMAGE"]
        for section, prompt in section_prompts.items():
            if section in session["reports"]:
                continue
            user_content = []
            if text:
                user_content.append({"type": "input_text", "text": text})
            for img in images:
                user_content.append({"type": "input_image", "image_url": img.s3_url})
            user_content.append({"type": "input_text", "text": prompt})
            openai_input = [
                {"role": "user", "content": user_content}
            ]
            openai_kwargs = {"model": "gpt-4o", "input": openai_input}
            try:
                if not OPENAI_API_KEY:
                    raise Exception("No OpenAI key")
                response_json = call_openai_responses_api(OPENAI_API_KEY, openai_kwargs)
                output_text = response_json["output"][0]["content"][0]["text"]
                session["reports"][section] = output_text
            except Exception as e:
                logger.warning(f"OpenAI failed for full report section {section}: {e}. Falling back to Gemini.")
                if not GEMINI_API_KEY:
                    raise HTTPException(status_code=503, detail="No AI service available.")
                client = get_gemini_client()
                gemini_parts = []
                if text:
                    gemini_parts.append(Part.from_text(text=text))
                for img in images:
                    gemini_parts.append(Part.from_uri(file_uri=img.s3_url, mime_type="image/png"))
                gemini_parts.append(Part.from_text(text=prompt))
                content = Content(role="user", parts=gemini_parts)
                response = client.models.generate_content(
                    model="models/gemini-2.5-flash-latest",
                    contents=[content],
                )
                output_text = ""
                if response.candidates and response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
                    parts = response.candidates[0].content.parts
                    if parts and len(parts) > 0:
                        first_part = parts[0]
                        output_text = getattr(first_part, 'text', "") or ""
                session["reports"][section] = output_text
        return FullReportResponse(
            id=str(uuid.uuid4()),
            project_name=request.project_name,
            qsa_name=request.qsa_name,
            date=request.date,
            scope_document={
                "id": str(uuid.uuid4()),
                **session["reports"]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full report error: {e}")
        raise HTTPException(status_code=500, detail=f"Full report error: {e}")

# NOTE: To use Gemini 2.5 Flash, install the official SDK:
# pip install google-generativeai
# NOTE: To use OpenAI Responses API, you may need to use the REST API if your SDK version does not support it.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app) #, host="0.0.0.0", port=8000) 