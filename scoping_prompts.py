"""
The file contains prompts for PCI_DSS audit scoping questions on 
network diagrams, dataflow diagrams, and asset inventory documents.
"""

objective_prompt = f"""
###Title: Objective

You are an expert PCI DSS auditor. From the given diagrams and information:
- Define the scope of the PCI DSS assessment for [Client Company Name], 
- Identify all assets, processes, and systems that store, process, or transmit cardholder data (CHD) and sensitive authentication data (SAD).
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
business_overview_prompt = f"""
###Title: Business Overview
You are an expert PCI DSS auditor. From the given diagrams and information:
- Identify Business Type: [e.g., E-commerce, Retail, Financial Services]
- Identify all Payment Channels: [List of channels, e.g., Online Payments, In-store POS, MobilePayments]
- Identify Compliance Requirements: PCI DSS [version], applicable to [specific goals].
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
cde_prompt = f"""
In-Scope Systems and Assets
You are an expert PCI DSS auditor. From the given diagrams and information, identify, analyze, and provide information for the CDE.
###Title: Cardholder Data Environment
- Identify Network Segments: [e.g., Production Network, DMZ]
- Systems
- POS Devices
- Payment Gateway
- E-commerce Platform
- Databases: [List of databases storing cardholder data]
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
connect_sys_prompt = f"""
In-Scope Systems and Assets
You are an expert PCI DSS auditor. From the given diagrams and information, identify, analyze and provide information for the:
###Title: Connected Systems
- Systems that interact with or support the CDE:
- Active Directory Servers
- Monitoring Tools
- Backup Systems
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
third_party_prompt = f"""
In-Scope Systems and Assets
###Title: Third-Party Services
You are an expert PCI DSS auditor. From the given diagrams and information, identify, analyze and provide information for the third-party services:
[e.g., Payment Processors, Cloud Hosting Providers]

If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
oof_sys_prompt = f"""
###Title: Out-of-Scope Systems
You are an expert PCI DSS auditor. From the given diagrams and information, identify, analyze and provide information for the out-of-scope systems:
Examples:
- Systems and processes excluded due to segmentation or unrelated to cardholder data.
- Corporate Email Systems
- HR Management Platforms
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
data_flow_prompt = f"""
###Title: Data Flows
You are an expert PCI DSS auditor. From the given diagrams and information, identify, analyze and provide information for the:
Cardholder Data Flows: Diagrams and descriptions showing the movement of cardholderdata across systems.
Example:
Customer -> POS Terminal -> Payment Gateway -> Processor

If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
risk_asmt_prompt = f"""
###Title: Risk Assessment Summary
You are an expert PCI DSS auditor. From the given diagrams and information about the network, create a risk assement summary.
- Identified risks for in-scope systems and mitigation plans.
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
asmp_exc_prompt = f"""
###Title: Assumptions and Exclusions
You are an expert PCI DSS auditor. From the given diagrams and information about the network, identify, analyze and provide information for the:
- Key assumptions: [e.g., Segmentation is effective for out-of-scope systems]
- Exclusions: [Specific systems or processes not handling CHD]
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
comp_val_prompt = f"""
###Title: Compliance Validation Approach
You are an expert PCI DSS auditor. From the given diagrams and information about the network, identify, analyze and provide information for the:
Methods to validate compliance:
- Onsite assessments
- Systematic testing
- Documentation review
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
roles_prompt = f"""
###Title: Stakeholders and Roles
You are an expert PCI DSS auditor. From the given diagrams and information about the network, identify the:
- Client Representative: [Client's point of contact]
- QSA/Auditor: [Assigned QSA Name]
- Technical Team: [List of technical representatives]
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request.
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""
nextstep_prompt = f"""
###Title: Next Steps
You are an expert PCI DSS auditor. From the given diagrams and information about the network, provide: 
- Timeline for completing assessment phases (scoping, assessment, remediation).
- Scheduled meetings or checkpoints.
Use and analyze the given information (Network diagram, Dataflow diagrams, and Inventory assement) to answer the request. 
If some required information is not present or you need additional information to perform the given requestion, list out the required information.

Format your response in Markdown as follows:
###Title
answer
Additional Information Required:
"""