## Background

Your team is building an internal knowledge assistant that helps employees query company documents and track file workflows. The system must combine conversational AI with structured data access, expose its capabilities as reusable tools, and run on Azure infrastructure.

You will build an AI agent that reasons about user intent, selects appropriate tools, and produces grounded, citable responses.

---

## Assignment: Azure AI Agent with RAG, Tool Use, and MCP

### Objective

Build a ReAct-based conversational agent using LangChain and LangGraph that can answer questions from ingested documents (RAG), query file statuses from a PostgreSQL database, and expose its tool capabilities through an MCP server — all deployed on Azure Container Apps.

---

## What You Need to Build

### 1) ReAct Agent (LangGraph)

Your agent should be able to:

- Implement a ReAct (Reason + Act) agent using LangGraph that decides when and which tools to call

- Maintain conversational context and handle multi-turn interactions (10+ turns)

- When no tool is needed, respond conversationally using the LLM directly

- Support configurable LLM providers (GLM/Z.AI for chat, Azure OpenAI for embeddings)

- Use structured output for tool invocations and responses

### 2) RAG Pipeline

The agent should:

- Accept document uploads (PDF, TXT, Markdown) and process them into chunks

- Generate embeddings using Azure OpenAI (`text-embedding-3-small`)

- Store vectors locally with FAISS (Phase 1), with a clear migration path to Azure AI Search (Phase 2)

- Retrieve relevant chunks for a user query using similarity search

- Generate grounded answers with source citations

- Expose RAG retrieval as an agent tool that the ReAct loop can invoke

### 3) PostgreSQL File Status Tool

You should implement:

- A PostgreSQL schema for tracking document/file workflow statuses (e.g., `pending`, `processing`, `approved`, `rejected`)

- An agent tool that queries this database to return file status information

- Support for lookup by filename, file ID, or status filter

- Graceful handling when no matching records are found

### 4) MCP Server

The agent's tools should be accessible externally:

- Expose agent tools (RAG retrieval, file status query, and any future tools) as an MCP (Model Context Protocol) server

- Any MCP-compatible client (Claude Desktop, VS Code, custom agents) should be able to discover and invoke the tools

- Implement proper tool descriptions and input schemas for MCP discovery

- Design for extensibility — adding a new tool should require minimal boilerplate

### 5) Evaluation Pipeline

The system should be measurable:

- Evaluate RAG quality: retrieval precision, recall, and answer faithfulness

- Evaluate agent behavior: tool selection accuracy, response appropriateness

- Create a test dataset of at least 15 queries covering all interaction types (RAG, DB query, multi-tool, conversational)

- Expose evaluation results via an API endpoint

### 6) Observability & Tracing

The system should be observable:

- Trace agent reasoning steps, tool invocations, and LLM calls end-to-end

- Track latency, token usage, and error rates per tool and per request

- Use LangSmith or Azure Application Insights for trace collection

- Implement structured logging across all components

---

## Example Interactions

**Document Question (RAG Tool)**
```
User: What is our company's policy on remote work?

Agent Reasoning: The user is asking about company policy. I should
search the document knowledge base.

Agent Action: invoke rag_retrieve("company policy remote work")

Agent: According to the Employee Handbook (Section 4.2), employees
may work remotely up to 3 days per week with manager approval.
Remote work arrangements must be documented through HR within
5 business days.
[Source: employee_handbook.pdf, chunk 23, score: 0.94]
```

**File Status Query (PostgreSQL Tool)**
```
User: What's the status of the Q4 budget report?

Agent Reasoning: The user wants to know a file's workflow status.
I should query the file status database.

Agent Action: invoke get_file_status(filename="Q4 budget report")

Agent: The Q4 Budget Report (ID: DOC-2024-0847) is currently in
"pending_review" status. It was submitted by Sarah Chen on Feb 10
and is assigned to the Finance team for review.
```

**Mixed Query (Multi-Tool Reasoning)**
```
User: Is the onboarding guide approved? And what does it say about
first-week expectations?

Agent Reasoning: This requires two steps. First, check the file
status. Then, retrieve content from the knowledge base.

Agent Action 1: invoke get_file_status(filename="onboarding guide")
Agent Action 2: invoke rag_retrieve("onboarding first-week expectations")

Agent: The Onboarding Guide (DOC-2024-0312) has been approved as of
Jan 15. Regarding first-week expectations, the guide states that new
hires should complete IT setup on Day 1, attend orientation on Day 2,
and meet with their assigned buddy by Day 3.
[Source: onboarding_guide.pdf, chunk 8, score: 0.91]
```

**General Conversation (No Tools)**
```
User: Thanks, that's really helpful!

Agent Reasoning: This is conversational. No tool needed.

Agent: You're welcome! Let me know if you have any other questions
about documents or file statuses.
```

---

## Technical Requirements

- **Agent Framework**: LangChain + LangGraph (ReAct pattern with tool nodes)

- **Chat LLM**: GLM (Z.AI) via OpenAI-compatible API (configurable provider)

- **Embeddings**: Azure OpenAI `text-embedding-3-small`

- **Vector Store**: FAISS locally, with Azure AI Search migration path

- **Database**: PostgreSQL for file status tracking

- **MCP**: Model Context Protocol SDK for tool server

- **Deployment**: Azure Container Apps with managed identity

- **API Framework**: FastAPI with async endpoints

- **Testing**: pytest with >70% coverage on core logic

- **Observability**: LangSmith or Azure Application Insights

- Plans should use structured output (JSON / Pydantic models)

- Must handle 10+ conversation turns without losing context

---

## What We'll Evaluate

| Area | Points | Criteria |
|------|--------|----------|
| **Agent Architecture** | 20 | LangGraph ReAct implementation, state management, clean tool routing |
| **RAG Pipeline** | 20 | Chunking strategy, retrieval quality, citation accuracy, grounded answers |
| **Tool Design & MCP** | 15 | Clean tool interfaces, proper MCP schema definitions, extensibility |
| **Database Integration** | 10 | PostgreSQL schema design, query tool robustness, error handling |
| **Evaluation Framework** | 15 | Metric coverage (retrieval + agent), test dataset quality, actionable results |
| **Observability** | 10 | End-to-end tracing, structured logs, meaningful metrics |
| **Code Quality & Tests** | 10 | Clean architecture, type hints, tests, documentation |
| **Total** | **100** | |

---

## Deliverables

1. Complete agent implementation (LangGraph + tools)

2. MCP server running and connectable by an MCP client

3. RAG pipeline with document ingestion and retrieval

4. PostgreSQL integration with file status tracking

5. Evaluation report showing RAG and agent metrics on the test dataset

6. UI implementation (CLI or Web — your preference)

7. README explaining: setup instructions, architecture diagram, design decisions

8. Requirements file (`requirements.txt` or `pyproject.toml`)

Make sure you build an agent (not a predefined workflow). The agent should reason about tool selection dynamically.
Use tools and structured output throughout.
Add tests and look for bugs.
Follow coding standards and fix code quality issues.

---

## Suggested Timeline

Week 1: RAG pipeline + LangGraph ReAct agent working locally
Week 2: PostgreSQL file status tool + MCP server
Week 3: Azure AI Search migration + evaluation pipeline
Week 4: Observability + Azure deployment + documentation
