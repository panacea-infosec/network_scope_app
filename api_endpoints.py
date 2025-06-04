import os
import base64
import uuid
import tempfile
import re
import shutil # For rmtree
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import fitz # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from scoping_prompts import (
    objective_prompt, business_overview_prompt, cde_prompt, connect_sys_prompt,
    third_party_prompt, oof_sys_prompt, data_flow_prompt, risk_asmt_prompt
)

app = FastAPI(title="PCI DSS Audit Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Initialize OpenAI client
# IMPORTANT: Set the OPENAI_API_KEY environment variable.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set. OpenAI API calls will fail.")
    
client = OpenAI(api_key=openai_api_key)

# --- Pydantic Models ---

class ProcessedImage(BaseModel):
    mime_type: str
    base64_data: str

class DocumentOutput(BaseModel):
    id: str # Unique ID for the processed document/item (e.g., UUID)
    original_name: str
    type: str # e.g., "PDF", "EXCELSHEET", "IMAGE", "UNKNOWN"
    text: Optional[str] = None
    # Images directly associated with this document item.
    # For type "IMAGE", this list contains the image itself (base64 encoded).
    # For type "PDF", this list is empty (images extracted from PDF are separate DocumentOutput items).
    images: List[ProcessedImage] = []
    parent_document_id: Optional[str] = None # ID of parent, if this item was extracted (e.g. image from PDF)

class ProcessFilesResponse(BaseModel):
    combined_text: str # All text from all documents, formatted with headers for context
    combined_images: List[ProcessedImage] # All images from all documents, for overall GPT context
    documents: List[DocumentOutput] # Detailed breakdown of each processed file/item

class ReportSectionRequest(BaseModel):
    section_name: str
    # Client sends back the context obtained from /process_files/
    combined_text: str
    combined_images: List[ProcessedImage]

class ReportSectionResponse(BaseModel):
    section_name: str
    content: str

class FullReportRequest(BaseModel):
    project_name: str
    qsa_name: str
    date: str # e.g., "YYYY-MM-DD"
    # Client sends back the context
    combined_text: str
    combined_images: List[ProcessedImage]

class FullReportResponse(BaseModel):
    id: str # Report ID (UUID)
    project_name: str
    qsa_name: str
    date: str
    scope_document: Dict[str, str] # Section name -> content mapping

class GeneralChatRequest(BaseModel):
    message: str
    # General context from all processed files, sent by client
    context_text: str
    context_images: List[ProcessedImage]
    history: List[Tuple[str, str]] # List of (user_query, assistant_response) tuples

class DocumentChatRequest(BaseModel):
    message: str
    document_name: str # Original name of the document being chatted about (for system prompt)
    # Context specific to this document, sent by client
    document_text: str
    document_images: List[ProcessedImage]
    history: List[Tuple[str, str]]

class ChatResponse(BaseModel):
    response: str

# --- Utility File Processing Functions ---

def process_image_to_base64(file_path: str) -> Tuple[str, str]:
    """Processes an image file to its MIME type and base64 encoded string."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        else:
            # Fallback for other types, or raise an error for unsupported image types
            # For simplicity, we assume common web image types.
            raise ValueError(f"Unsupported image type: {ext}")
        with open(file_path, "rb") as f:
            return mime_type, base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing image {os.path.basename(file_path)}: {e}")

def process_excel_to_text(file_path: str) -> str:
    """Extracts text from an Excel file, combining all sheets."""
    try:
        with pd.ExcelFile(file_path) as xls: # Ensures xls.close() is called
            excel_content_parts = []
            for name in xls.sheet_names:
                # Adding error handling for individual sheet parsing
                try:
                    sheet_data = xls.parse(name).to_string()
                    excel_content_parts.append(f"--- Sheet: {name} ---\n{sheet_data}")
                except Exception as sheet_error:
                    excel_content_parts.append(f"--- Sheet: {name} ---\nError parsing sheet: {sheet_error}")
            return "\n\n".join(excel_content_parts)
    except Exception as e:
        # This will catch errors like file not found, or not a valid Excel file
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing Excel file {os.path.basename(file_path)}: {e}")

def process_pdf_to_text_and_images(file_path: str, image_output_dir: str) -> Tuple[str, List[str]]:
    """
    Extracts full text from a PDF and saves images larger than 100x100 pixels.
    Returns the extracted text and a list of paths to the saved images.
    """
    doc = None # Initialize doc to None for robust error handling in finally block
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n" # Add double newline for separation

        extracted_image_paths = []
        pdf_base_name = re.sub(r'[^\w\.-]', '_', os.path.splitext(os.path.basename(file_path))[0])

        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Handle potential non-RGB/RGBA images if necessary (e.g. CMYK)
                if pix.n >= 4 and not pix.alpha: # Example: convert CMYK to RGB
                     pix = fitz.Pixmap(fitz.csRGB, pix)

                if pix.width > 100 and pix.height > 100:
                    image_filename = f"extracted_from_{pdf_base_name}_p{page_num}_i{img_index}_{xref}.png"
                    full_image_path = os.path.join(image_output_dir, image_filename)
                    pix.save(full_image_path)
                    extracted_image_paths.append(full_image_path)
                pix = None  # Dereference Pixmap object to free memory
        return full_text, extracted_image_paths
    except Exception as e:
        error_detail = f"Error processing PDF file {os.path.basename(file_path)}: {e}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)
    finally:
        if doc:
            doc.close()


# --- OpenAI GPT-4o Analysis Functions ---

def _call_openai_api(messages: List[Dict], max_tokens: int, temperature: float) -> str:
    """Helper function to call OpenAI API and handle common exceptions."""
    if not openai_api_key: # Check if API key is available
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="OpenAI API key not configured. Cannot process request.")
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or your preferred model
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e: # Catch specific OpenAI errors if possible
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OpenAI API error: {e}")


def analyze_with_gpt4o_report_generation(
    prompt_for_llm: str,
    context_text_for_llm: str,
    context_images_for_llm: List[ProcessedImage]
) -> str:
    """Sends content to GPT-4o for report section generation."""
    content_parts = [{"type": "text", "text": f"Context from document(s):\n{context_text_for_llm}\n\nTask: {prompt_for_llm}"}]
    for img_model in context_images_for_llm:
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_model.mime_type};base64,{img_model.base64_data}"}})
    
    messages = [{"role": "user", "content": content_parts}]
    return _call_openai_api(messages, max_tokens=2000, temperature=0.3)


def analyze_with_gpt4o_chat(
    user_message_text: str,
    context_text_for_llm: str,
    context_images_for_llm: List[ProcessedImage],
    chat_history_tuples: List[Tuple[str, str]],
    system_prompt: str
) -> str:
    """Sends chat content to GPT-4o and returns the model's response."""
    messages = [{"role": "system", "content": system_prompt}]

    for user_msg, assistant_msg in chat_history_tuples:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    current_user_content_parts = []
    context_block = ""
    if context_text_for_llm or context_images_for_llm:
        context_block = "Relevant Context from Document(s):\n"
        if context_text_for_llm: # Ensure context_text is not empty before adding
            context_block += f"{context_text_for_llm}\n\n"
    
    final_user_text_prompt = f"{context_block}User Question: {user_message_text}"
    current_user_content_parts.append({"type": "text", "text": final_user_text_prompt})

    for img_model in context_images_for_llm:
        current_user_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_model.mime_type};base64,{img_model.base64_data}"}})
    
    messages.append({"role": "user", "content": current_user_content_parts})
    return _call_openai_api(messages, max_tokens=1500, temperature=0.5)


# --- API Endpoints ---

@app.post("/process_files/", response_model=ProcessFilesResponse)
async def process_files_endpoint(files: List[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    
    # Stores details of each processed item (original file or extracted image)
    # Each item dict: {id, original_name, temp_file_path, text, images_for_gpt, type, parent_document_id}
    processed_item_details = [] 

    try:
        for uploaded_file in files:
            original_fn = uploaded_file.filename
            # Sanitize filename for saving, use UUID for unique doc ID later
            cleaned_base_fn = re.sub(r'[^\w\.-]', '_', os.path.splitext(original_fn)[0])
            file_extension = os.path.splitext(original_fn)[1].lower()
            
            # Create a unique path in temp_dir for the uploaded file
            temp_file_id = str(uuid.uuid4()) # Unique ID for the temp file itself
            temp_file_path = os.path.join(temp_dir, f"{cleaned_base_fn}_{temp_file_id}{file_extension}")
            
            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
            
            # This ID is for the DocumentOutput object representing this uploaded file
            file_item_id = str(uuid.uuid4()) 
            item_detail = {
                "id": file_item_id,
                "original_name": original_fn,
                "text": "",
                "images_for_gpt": [], # List[ProcessedImage] for this specific item
                "type": "UNKNOWN", # Default type
                "parent_document_id": None
            }

            if file_extension in ['.png', '.jpg', '.jpeg']:
                item_detail["type"] = "IMAGE"
                mime, b64 = process_image_to_base64(temp_file_path)
                item_detail["images_for_gpt"].append(ProcessedImage(mime_type=mime, base64_data=b64))
            elif file_extension in ['.xlsx', '.xls']:
                item_detail["type"] = "EXCELSHEET"
                item_detail["text"] = process_excel_to_text(temp_file_path)
            elif file_extension == '.pdf':
                item_detail["type"] = "PDF"
                pdf_text, extracted_image_paths = process_pdf_to_text_and_images(temp_file_path, temp_dir)
                item_detail["text"] = pdf_text
                
                # Process images extracted from this PDF
                for ext_img_path in extracted_image_paths:
                    ext_img_id = str(uuid.uuid4())
                    # Use the generated filename which is already somewhat descriptive
                    ext_img_original_name = os.path.basename(ext_img_path) 
                    ext_mime, ext_b64 = process_image_to_base64(ext_img_path)
                    processed_item_details.append({
                        "id": ext_img_id,
                        "original_name": f"{ext_img_original_name} (from {original_fn})",
                        "text": "", 
                        "images_for_gpt": [ProcessedImage(mime_type=ext_mime, base64_data=ext_b64)],
                        "type": "IMAGE",
                        "parent_document_id": file_item_id # Link to the parent PDF's item ID
                    })
            else:
                item_detail["text"] = f"File type '{file_extension}' not supported for deep content extraction."

            processed_item_details.append(item_detail)

        # --- Construct the final response ---
        response_documents_output = []
        combined_text_parts = []
        all_processed_images_for_gpt = []

        for detail in processed_item_details:
            doc_output = DocumentOutput(
                id=detail["id"],
                original_name=detail["original_name"],
                type=detail["type"],
                text=detail["text"] if detail["text"] else None,
                images=detail["images_for_gpt"],
                parent_document_id=detail["parent_document_id"]
            )
            response_documents_output.append(doc_output)

            if detail["text"]:
                combined_text_parts.append(f"--- Document: {detail['original_name']} (Type: {detail['type']}) ---\n{detail['text']}\n")
            
            all_processed_images_for_gpt.extend(detail["images_for_gpt"])
        
        final_combined_text = "\n".join(combined_text_parts)
        
        # Deduplicate images in all_processed_images_for_gpt if needed (e.g., if a standalone image was also part of a PDF)
        # Simple deduplication based on base64 content
        seen_b64_hashes = set()
        unique_combined_images = []
        for img in all_processed_images_for_gpt:
            if img.base64_data not in seen_b64_hashes:
                unique_combined_images.append(img)
                seen_b64_hashes.add(img.base64_data)
        
        return ProcessFilesResponse(
            combined_text=final_combined_text,
            combined_images=unique_combined_images,
            documents=response_documents_output
        )

    except HTTPException:
        raise # Re-raise HTTPExceptions from utility functions
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error in file processing: {str(e)}")
    finally:
        shutil.rmtree(temp_dir) # Ensure temporary directory is always cleaned up

# --- Report Generation Endpoints ---
SECTION_PROMPTS_MAP = {
    "Objective": objective_prompt,
    "Business Overview": business_overview_prompt,
    "Cardholder Data Environment": cde_prompt,
    "Connected Systems": connect_sys_prompt,
    "Third Parties": third_party_prompt,
    "Out-of-Scope Systems": oof_sys_prompt,
    "Data Flows": data_flow_prompt,
    "Risk Assessment": risk_asmt_prompt,
}

@app.post("/generate_report_section/", response_model=ReportSectionResponse)
async def generate_report_section(request: ReportSectionRequest):
    if request.section_name not in SECTION_PROMPTS_MAP:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid report section name: {request.section_name}. Valid names are: {list(SECTION_PROMPTS_MAP.keys())}")
    
    prompt_text = SECTION_PROMPTS_MAP[request.section_name]
    content = analyze_with_gpt4o_report_generation(
        prompt_for_llm=prompt_text,
        context_text_for_llm=request.combined_text,
        context_images_for_llm=request.combined_images
    )
    return ReportSectionResponse(section_name=request.section_name, content=content)

@app.post("/generate_full_report/", response_model=FullReportResponse)
async def generate_full_report(request: FullReportRequest):
    scope_document_content = {}
    for section_name, prompt_text in SECTION_PROMPTS_MAP.items():
        content = analyze_with_gpt4o_report_generation(
            prompt_for_llm=prompt_text,
            context_text_for_llm=request.combined_text,
            context_images_for_llm=request.combined_images
        )
        scope_document_content[section_name] = content
    
    report_id = str(uuid.uuid4())
    return FullReportResponse(
        id=report_id,
        project_name=request.project_name,
        qsa_name=request.qsa_name,
        date=request.date,
        scope_document=scope_document_content
    )

# --- Chat Endpoints ---

@app.post("/chat/general/", response_model=ChatResponse)
async def handle_general_chat(request: GeneralChatRequest):
    system_prompt = (
        "You are an expert PCI DSS auditor. "
        "Answer questions based on the provided context from all uploaded documents. "
        "The context contains text extracted from various files, each prefixed with its filename and type. "
        "If the user's question is irrelevant to PCI DSS or the provided documents, "
        "politely guide them back to the topic."
    )
    response_text = analyze_with_gpt4o_chat(
        user_message_text=request.message,
        context_text_for_llm=request.context_text,
        context_images_for_llm=request.context_images,
        chat_history_tuples=request.history,
        system_prompt=system_prompt
    )
    return ChatResponse(response=response_text)

@app.post("/chat/document/", response_model=ChatResponse)
async def handle_document_chat(request: DocumentChatRequest):
    system_prompt = f"""You are an expert PCI DSS auditor.
You will be answering questions specifically about the document named '{request.document_name}'.
The content (text and images) for this document is provided in the current context.
Scan everything present in the document's provided content: figures, nodes, boxes, text, lines, etc.
Note all components and their relationships if applicable to the question.
Do not add any information that is not present in the provided document content for '{request.document_name}'.
Only use information from this specific document's content to answer the user's question.
"""
    response_text = analyze_with_gpt4o_chat(
        user_message_text=request.message,
        context_text_for_llm=request.document_text,
        context_images_for_llm=request.document_images,
        chat_history_tuples=request.history,
        system_prompt=system_prompt
    )
    return ChatResponse(response=response_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app) # , host="127.0.0.1", port=8000