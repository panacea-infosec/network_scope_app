## PCI-DSS Assist App
### Application to create/draft a PCI-DSS scope document using the given documents, along with a chat assistant.
### Files:
1. fastapi_endpoints.py: Creates a content ID 'content_hash' for the processed files, i.e., all extracted text and images in base64 strings. Need to pass 'content_hash' to functions to point to processed content.
2. api_endpoints.py: Completely stateless, with no sessions or content IDs that need to be passed; instead, the endpoint functions require the processed content itself to be passed.
3. fast-api_scope_pci.py: Need to create a session and use the session ID to pass information to the endpoint functions.
4. pci_asst_app.py: Gradio app demonstrating the scope generation and chat assistant application.
