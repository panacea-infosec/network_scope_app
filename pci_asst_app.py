import gradio as gr
import asyncio
import os
import base64
from openai import OpenAI
import pandas as pd
import pdf2image
from PyPDF2 import PdfReader
from PIL import Image
import openpyxl
import fitz  
from scoping_prompts import (
    objective_prompt, business_overview_prompt, cde_prompt, connect_sys_prompt,
    third_party_prompt, oof_sys_prompt, data_flow_prompt, risk_asmt_prompt,
    asmp_exc_prompt, comp_val_prompt, roles_prompt, nextstep_prompt
)

client = OpenAI(api_key="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")

# File processing functions (same as original code)
def process_image(file_path: str) -> tuple[str, str]:
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

def process_pdf(file_path: str) -> str:
    try:
        return "\n".join([page.extract_text() for page in PdfReader(file_path).pages if page.extract_text()])
    except Exception as e:
        return f"Error processing PDF: {e}"

def process_files(file_paths: list) -> tuple[str, list]:
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
            text += process_pdf(path) + "\n"
    return text, images


# analysis functions
def analyze_with_gpt4o(prompt: str, text: str, images: list, history=None) -> str:
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


# Gradio UI Components
prompts = {
    objective_prompt: "Objective",
    business_overview_prompt: "Business Overview",
    cde_prompt: "Cardholder Data Environment",
    connect_sys_prompt: "Connected Systems",
    third_party_prompt: "Third Parties",
    oof_sys_prompt: "Out-of-Scope",
    data_flow_prompt: "Data Flows",
    risk_asmt_prompt: "Risk Assessment",
    asmp_exc_prompt: "Assumptions/Exclusions",
    comp_val_prompt: "Compliance Validation",
    roles_prompt: "Stakeholders",
    nextstep_prompt: "Next Steps"
}

# def create_prompt_handler(prompt_text, heading, index):
#     def handler(processed_data, progress=gr.Progress()):
#         text, images = processed_data
#         progress(0, desc=f"Generating {heading}...")
#         response = analyze_with_gpt4o(prompt_text, text, images)
#         progress(1.0)
#         # Create a list of updates for all response boxes
#         updates = [gr.update(visible=i == index, value=response if i == index else None) 
#                  for i in range(len(prompts))]
#         return updates
#     return handler
##NOTE
# Modify the create_prompt_handler function
def create_prompt_handler(prompt_text, heading, index):
    def handler(processed_data, reports):
        progress = gr.Progress()
        progress(0, desc=f"Generating {heading}...")
        # Check if report already exists
        if heading in reports:
            return [gr.update(visible=i == index, value=reports[heading] if i == index else None) 
                   for i in range(len(prompts))]
        
        # Generate new report if not cached
        text, images = processed_data
        # progress(0, desc=f"Generating {heading}...")
        response = analyze_with_gpt4o(prompt_text, text, images)
        progress(1.0)
        
        # Update reports cache
        reports[heading] = response
        
        return [gr.update(visible=i == index, value=response if i == index else None) 
               for i in range(len(prompts))]
    return handler


# Gradio Application
with gr.Blocks(title="Document Analyzer") as app:
    processed_data = gr.State(("", []))  # For main analysis
    chat_history = gr.State([])
    document_data = gr.State({})  # Stores {filename: (text, images)}
    doc_chat_histories = gr.State({})  # Stores {filename: chat_history}
    gallery_data = gr.State([])  # Stores gallery items for both tabs
    ##NOTE:
    reports_state = gr.State({})  # to store generated reports


    with gr.Tabs():
        with gr.Tab("Analysis Report"):
            #  main analysis components
            with gr.Row():
                with gr.Column(scale=1):
                    # Call the dummy function once on app launch.
                    files = gr.File(file_count="multiple", file_types=[".pdf", ".xlsx", ".png", ".jpg", ".jpeg"])
                    process_btn = gr.Button("Process Files", variant="primary")
                    progress_bar = gr.Progress()
                with gr.Column(scale=14):
                    analysis_gallery = gr.Gallery(label="Uploaded Files", columns=4, height="auto")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Analysis")
                    prompt_btns = [gr.Button(heading) for heading in prompts.values()]
                
                with gr.Column(scale=3):
                    gr.Markdown("## Reports")
                    response_boxes = [gr.Markdown(visible=False, label=heading) for heading in prompts.values()]
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Chat with Assistant")
                    chatbot = gr.Chatbot(height=200)
                    chat_input = gr.Textbox(placeholder="Ask about the documents...")
                    chat_btn = gr.Button("Send")

        with gr.Tab("Document Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Select Document")
                    doc_selector = gr.Dropdown(label="Uploaded Documents", interactive=True, choices=[])
                    gr.Markdown("## Document Chat")
                    doc_chatbot = gr.Chatbot(height=400)
                    doc_chat_input = gr.Textbox(placeholder="Ask about this specific document...")
                    doc_chat_btn = gr.Button("Send")

                with gr.Column(scale=8):
                    image_output = gr.Image(label="Image Viewer")
                    pdf_output = gr.HTML(label="PDF Viewer")

    def generate_gallery_items(file_paths):
        gallery_items = []
        for path in file_paths:
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # For images, use the file directly
                gallery_items.append(path)
            # elif path.lower().endswith('.pdf'):
            #     # For PDFs, generate a thumbnail
            #     try:
            #         from pdf2image import convert_from_path
            #         images = convert_from_path(path, first_page=1, last_page=1)
            #         if images:
            #             # Save first page as temp image
            #             import tempfile
            #             with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            #                 images[0].save(tmp.name, "JPEG")
            #                 gallery_items.append(tmp.name)
            #     except ImportError:
            #         gallery_items.append("pdf_icon.png")  # Fallback icon
            # else:
            #     # For other files, use a generic icon
            #     gallery_items.append("file_icon.png")
        return gallery_items


    # File Processing Logic
    def process_uploaded_files(files, progress=gr.Progress()):
        file_data = {}
        file_paths = [f.name for f in files]
        progress(0, desc="Starting processing...")
        
        gallery_items = generate_gallery_items(file_paths)
        
        # Process individual files
        for i, path in enumerate(file_paths):
            progress(i/len(file_paths), f"Processing {os.path.basename(path)}")
            filename = os.path.basename(path)
            text, images = "", []
            
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(process_image(path))
            elif path.lower().endswith(('.xlsx', '.xls')):
                text = process_excel(path)
            elif path.lower().endswith('.pdf'):
                text = process_pdf(path)
            
            file_data[filename] = (text, images)
        
        combined_text, combined_images = process_files(file_paths)
        progress(1.0, "Processing complete!")
        return (
            file_data,  # document_data
            (combined_text, combined_images),  # processed_data
            gr.update(choices=list(file_data.keys())),  # doc_selector
            gallery_items,  # analysis_gallery
            {}  # Reset reports_state ##NOTE
        )
    
    # Document Chat Handling
    def handle_doc_chat(message, history, filename, document_data, doc_chat_histories):
        if not filename:
            return history + [(message, "Please select a document first!")], doc_chat_histories
        
        # Get document-specific data
        text, images = document_data.get(filename, ("", []))
        
        # Get or create chat history
        file_history = doc_chat_histories.get(filename, [])
        
        # Convert history to OpenAI format
        messages = [{
            "role": "system",
            "content":f"""
                You are an expert PCI DSS auditor. 
                Answers the user's question according to the file provided.
                Scan everything present in the file: figures, nodes, boxes, text, lines, etc.
                Note all components present in the network.
                Note all relationships between the components in the network.
                Do not add any information that is not present in the file originally.
                Only use information provided by the file to answer user's question.
                Analyze document: {filename}\nContent: {text}.
                """
        }]
        
        # Add previous conversation history
        for user_msg, bot_resp in file_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_resp})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = analyze_with_gpt4o(
            message, 
            text, 
            images, 
            history=messages  # Pass properly formatted messages
        )
        
        # Update histories
        new_history = file_history + [(message, response)]
        doc_chat_histories[filename] = new_history
        
        return new_history, doc_chat_histories

    # Update document preview
    def update_doc_preview(filename, document_data):
        if not filename or filename not in document_data:
            return "Select a document to preview"
        text, _ = document_data[filename]
        return f"## {filename}\n\n{text[:500]}{'...' if len(text) > 500 else ''}"
    
    def update_dropdown(files):
        filenames = [os.path.basename(file.name) for file in files] if files else []
        return gr.Dropdown(choices=filenames)

    def display_file(selected_file, uploaded_files):
        if not selected_file or not uploaded_files:
            return None, None
        
        # Find the selected file's path
        file_path = None
        for file in uploaded_files:
            if os.path.basename(file.name) == selected_file:
                file_path = file.name
                break
        
        if not file_path:
            return None, None
        
        # Determine file type and render accordingly
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            return file_path, None
        elif ext == '.pdf':
            with open(file_path, "rb") as f:
                pdf_data = f.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            pdf_html = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="800px"></iframe>'
            return None, pdf_html
        return None, None


    # Event handlers
    # process_btn.click(
    #     process_uploaded_files,
    #     inputs=[files],
    #     outputs=[document_data, processed_data, doc_selector, analysis_gallery, doc_chat_gallery],
    #     show_progress=True
    # )
    ##NOTE
    # Update the process_uploaded_files outputs in the click handler
    process_btn.click(
        process_uploaded_files,
        inputs=[files],
        outputs=[document_data, processed_data, doc_selector, analysis_gallery, reports_state],
        show_progress=True
    )
    
    doc_selector.change(
        display_file,
        inputs=[doc_selector, files],
        outputs=[image_output, pdf_output]
    )
    
    doc_chat_btn.click(
        handle_doc_chat,
        inputs=[doc_chat_input, doc_chatbot, doc_selector, document_data, doc_chat_histories],
        outputs=[doc_chatbot, doc_chat_histories]
    ).then(lambda: "", outputs=[doc_chat_input])

    doc_chat_input.submit(
        handle_doc_chat,
        inputs=[doc_chat_input, doc_chatbot, doc_selector, document_data, doc_chat_histories],
        outputs=[doc_chatbot, doc_chat_histories]
    ).then(lambda: "", outputs=[doc_chat_input])

    # prompt handling and main chat handling 
    # Prompt Handling
    # for index, (btn, (prompt, heading)) in enumerate(zip(prompt_btns, prompts.items())):
    #     btn.click(
    #         create_prompt_handler(prompt, heading, index),
    #         inputs=[processed_data],
    #         outputs=response_boxes,
    #         show_progress=True
    #     )
    ##NOTE
    # Update the click handlers
    for index, (btn, (prompt, heading)) in enumerate(zip(prompt_btns, prompts.items())):
        btn.click(
            create_prompt_handler(prompt, heading, index),
            inputs=[processed_data, reports_state],
            outputs=response_boxes,
            show_progress=True  # This enables the global progress bar
        ).then(
            lambda r: r,
            inputs=[reports_state],
            outputs=[reports_state],
            show_progress=False
        )
    
    # Chat Handling
    def handle_chat(message, history, data):
        progress = gr.Progress()  # Create a fresh instance each time
        progress(0, "Analyzing question...")
        text, images = data
        
        # Convert history to OpenAI format
        messages = []
        chat_prompt = f"""
        You are an expert PCI DSS auditor.
        The following information is extracted from several files. 
        Each file's content is preceded by a header indicating the file type and filename.
        For example:
        --- Excel File: assest_inventory.xlsx ---
        [Excel content here]
        --- PDF File: dataflow.pdf ---
        [PDF content here]
        --- Image File: screenshot.jpg ---
        Use these headers to distinguish between the documents and answer questions accordingly. 
        Answers the user's question according to the information provided.
        If user's question is irrelevant to the topic, 
        politely remind user to stick to asking questions about the given documents in a PCI DSS audit context.
        """
        messages.append({"role": "developer", "content": chat_prompt})
        for user, bot in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": bot})
        
        response = analyze_with_gpt4o(message, text, images, messages)
        progress(1.0)
        return history + [(message, response)]
    
    chat_btn.click(
        handle_chat,
        inputs=[chat_input, chat_history, processed_data],
        outputs=[chat_history]
        # show_progress=True
    ).then(
        lambda: "",
        outputs=[chat_input]
    )

    chat_input.submit(
        handle_chat,
        inputs=[chat_input, chat_history, processed_data],
        outputs=[chat_history]
        # show_progress=True
    ).then(
        lambda: "",
        outputs=[chat_input]
    )
    
    chat_history.change(
        lambda h: gr.update(value=h),
        inputs=[chat_history],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    app.launch() #share=True