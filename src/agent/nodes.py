from src.agent.state import AgentState

from langchain_core.messages import SystemMessage

def create_agent_node(llm_with_tools):
    system_message = SystemMessage(content="""You are a helpful AI assistant with access to specialized tools for document-related queries.

Available tools:
- rag_tool: Search document content. Use this for questions about what's INSIDE documents - policies, procedures, facts, numbers, guidelines, or any textual content.
- file_status_tool: Check document workflow status. Use this for questions about a document's STATE - approval status (approved/rejected/pending/processing), who submitted it, which department owns it, or submission dates.

Guidelines:
1. Analyze each question to determine which tool(s) are needed:
   - Content questions (what does it say, what are the rules, explain the policy) → rag_tool
   - Status questions (is it approved, who submitted it, what department) → file_status_tool
   - Combined questions (is it approved AND what does it say about X) → use BOTH tools

2. For multi-part questions, call each relevant tool separately and combine the results in your response.

3. When citing document content, include the source filename and relevant context.

4. If a tool returns no results, inform the user and suggest alternatives.

5. For general conversation unrelated to documents or file status, respond naturally without tools.

6. Be concise but thorough, clearly distinguishing between content information and status information in your responses.
    """)
    
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke([system_message] + state['messages'])
        return {"messages": [response]}
    
    return agent_node
