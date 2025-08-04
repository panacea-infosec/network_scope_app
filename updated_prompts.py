"""
The file contains prompts for PCI_DSS audit scoping questions on 
network diagrams, dataflow diagrams, and asset inventory documents.
Updated for single-generation approach.
"""

system_prompt = """You are an expert PCI DSS auditor with extensive experience in scope document creation. Your task is to create a comprehensive PCI DSS scope document by thoroughly analyzing all provided diagrams, documentation, and information.

KEY RESPONSIBILITIES:
- Analyze all uploaded documents including network diagrams, dataflow diagrams, asset inventories, and any extracted text
- Identify all assets, processes, and systems that store, process, or transmit cardholder data (CHD) and sensitive authentication data (SAD)
- Create a detailed scope document following PCI DSS requirements and best practices
- Provide specific analysis based on the provided information without making unfounded assumptions

ANALYSIS APPROACH:
1. First, carefully review ALL provided documents and images
2. Extract relevant information about systems, networks, data flows, and security controls
3. Identify in-scope vs out-of-scope systems based on CHD/SAD handling
4. Map data flows and system interconnections
5. Assess security controls and potential risks
6. Generate each section with specific details from your analysis
7. Do not make any assumptions or make up information to form any section of the report. If you need additional information, state it clearly for each section of the scope document report, under 'Additional Information Required'.

FORMATTING REQUIREMENTS:
- Use clear markdown formatting with appropriate headers
- Provide detailed, professional analysis for each section
- Include specific references to analyzed documents when possible
- Clearly state what additional information is required for each section of the scope document report, under 'Additional Information Required' for better analysis and response.
- Maintain consistency in terminology and technical details across sections

The scope document must include the following sections in order:
1. Objective
2. Business Overview
3. Cardholder Data Environment
4. Connected Systems
5. Third-Party Services
6. Out-of-Scope Systems
7. Data Flows
8. Risk Assessment Summary
9. Assumptions and Exclusions
"""
# 10. Compliance Validation Approach
# 11. Stakeholders and Roles
# 12. Next Steps

objective_prompt = """Define the comprehensive scope of the PCI DSS assessment, including:
- Primary assessment objectives and goals
- Scope boundaries and criteria for inclusion/exclusion
- All assets, processes, and systems that store, process, or transmit cardholder data (CHD) and sensitive authentication data (SAD)
- Assessment timeline and key milestones
- Compliance level and applicable PCI DSS version

Base your analysis on the provided network diagrams, asset inventories, and documentation. Be specific about which systems and processes are included in scope and why."""

business_overview_prompt = """Provide a comprehensive business overview covering:
- Business type and primary industry focus
- All payment channels and processing methods (e.g., online payments, in-store POS, mobile payments, phone orders)
- Payment card types accepted (Visa, MasterCard, American Express, etc.)
- Transaction volumes and processing patterns
- Geographic locations and operational scope
- Key business processes that handle payment data
- Regulatory and compliance requirements beyond PCI DSS

Analyze the provided documentation to identify business context and payment processing activities."""

cde_prompt = """Identify and analyze the Cardholder Data Environment (CDE) including:
- Network segments containing CHD/SAD (Production networks, DMZ, isolated segments)
- All systems storing, processing, or transmitting cardholder data:
  * POS devices and terminals
  * Payment gateways and processors
  * E-commerce platforms and web applications
  * Databases containing cardholder data
  * Application servers and middleware
  * File systems and data repositories
- Network infrastructure components (switches, routers, firewalls)
- Security controls protecting the CDE
- Data retention policies and storage locations

Use the network diagrams and asset inventories to provide specific system details and configurations."""

connect_sys_prompt = """Analyze and document systems that connect to or support the CDE:
- Authentication systems (Active Directory, LDAP servers)
- Monitoring and logging systems (SIEM, log collectors, monitoring tools)
- Backup and recovery systems
- Network management systems
- Administrative workstations and jump servers
- Security appliances (IDS/IPS, vulnerability scanners)
- Time synchronization servers (NTP)
- DNS and network services

For each connected system, describe its relationship to the CDE and security controls in place."""

third_party_prompt = """Identify and evaluate all third-party services and providers:
- Payment processors and acquirers
- Cloud hosting and infrastructure providers
- Software as a Service (SaaS) providers
- Managed service providers
- Network connectivity providers
- Certificate authorities
- Compliance and security service providers

For each third-party service, document:
- Services provided and data access levels
- PCI DSS compliance status
- Data sharing agreements and controls
- Integration points with internal systems"""

oof_sys_prompt = """Clearly define out-of-scope systems and justify their exclusion:
- Systems with no connection to CHD/SAD processing
- Properly segmented networks and systems
- Corporate systems (email, HR, financial systems unrelated to payments)
- Development and testing environments (if properly isolated)
- Guest networks and public systems

For each out-of-scope determination:
- Provide clear justification for exclusion
- Document segmentation controls
- Confirm no CHD/SAD access or processing capability
- Validate network isolation measures"""

data_flow_prompt = """Map comprehensive data flows showing cardholder data movement:
- Customer payment initiation points
- Data capture and entry systems
- Processing and authorization flows
- Data storage and retention points
- Data transmission paths and protocols
- Integration points between systems
- Data disposal and destruction processes

Create detailed flow descriptions such as:
- Customer → POS Terminal → Payment Gateway → Processor → Acquirer
- Online Customer → Web Application → Database → Payment Processor
- Phone Orders → Call Center System → CRM → Payment Gateway

Include data formats, encryption methods, and security controls at each step."""

risk_asmt_prompt = """Conduct a comprehensive risk assessment covering:
- Identified vulnerabilities and security gaps
- Network segmentation effectiveness
- Access control adequacy
- Data protection measures
- System hardening status
- Monitoring and detection capabilities
- Incident response preparedness
- Physical security controls

For each identified risk:
- Describe the potential impact
- Assess likelihood and severity
- Recommend specific mitigation strategies
- Prioritize remediation efforts"""



