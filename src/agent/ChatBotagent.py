import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith tracing

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
import functools
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
import uuid
from dotenv import load_dotenv

# Import existing agents from your files
from product_agent_graph import product_agent
from order_agent_graph import order_agent
from reviewer_reaAct_summary_agent import reviewer_reaAct_summary_agent

load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7
)

#-----------------------------------------------------------------------------
# State for the orchestrator chatbot
class ChatBotState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    current_agent: str
    is_done: bool  # Flag to indicate user is done chatting

#-----------------------------------------------------------------------------
# Helper function to invoke an agent (reusable pattern with functools.partial)
def agent_node(state, agent, name, config):
    """
    Generic agent node that can wrap any agent.
    Uses functools.partial to pre-fill agent and name parameters.
    """
    # Extract thread-id from config for conversation memory
    thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
    
    # Set the config for calling the agent
    agent_config = {"configurable": {"thread_id": thread_id}}
    
    # Prepare state for the agent
    agent_state = {"messages": state["messages"]}
    
    # Add extra fields for summary agent if needed
    if name == "Summary_Agent":
        agent_state["retry_count"] = 0
        agent_state["feedback"] = ""
    
    # Invoke the agent with the state
    result = agent.invoke(agent_state, agent_config)
    
    # Convert the agent output into a format suitable for global state
    if isinstance(result, ToolMessage):
        final_result = result
    else:
        final_result = AIMessage(content=result['messages'][-1].content)
    
    return {
        "messages": [final_result],
        "current_agent": name
    }

#-----------------------------------------------------------------------------
# Create agent nodes using functools.partial
product_node = functools.partial(
    agent_node,
    agent=product_agent.agent_graph,
    name="Product_Agent"
)

order_node = functools.partial(
    agent_node,
    agent=order_agent.agent_graph,
    name="Order_Agent"
)

# Summary node - uses the existing reviewer_reaAct_summary_agent
summary_node = functools.partial(
    agent_node,
    agent=reviewer_reaAct_summary_agent.agent_graph,
    name="Summary_Agent"
)

#-----------------------------------------------------------------------------
# Orchestrator Chatbot that routes to existing agents
class ChatBotOrchestrator:
    
    def __init__(self, model, debug=False):
        self.debug = debug
        self.model = model
        
        # Router prompt - includes "done" detection
        self.router_prompt = """
        You are a routing assistant. Analyze the user's message and determine the intent:
        - "product" - Questions about laptop prices, features, specifications, or product information
        - "order" - Questions about orders, order status, updating quantities, or order management
        - "general" - General greetings, small talk, or unclear intent
        - "done" - User is ending the conversation (goodbye, thanks bye, I'm done, quit, exit, that's all, etc.)
        
        Respond with ONLY one word: product, order, general, or done
        """
        
        self.general_prompt = """
        You are a friendly and professional customer service chatbot for a laptop company.
        Handle greetings and small talk professionally.
        Guide users to ask about products or orders if they seem unsure.
        """
        
        # Build the orchestrator graph
        graph = StateGraph(ChatBotState)
        
        # Add nodes - using the pre-built partial functions for agents
        graph.add_node("router", self.route_query)
        graph.add_node("product_agent", product_node)
        graph.add_node("order_agent", order_node)
        graph.add_node("general_agent", self.handle_general)
        graph.add_node("summarize_agent", summary_node)  # Summary as a graph node
        
        # Set entry point
        graph.add_edge(START, "router")
        
        # Router conditional edges - includes "done" routing to summary
        graph.add_conditional_edges(
            "router",
            self.get_next_node,
            {
                "product": "product_agent",
                "order": "order_agent",
                "general": "general_agent",
                "done": "summarize_agent"  # Routes to summary when user is done
            }
        )
        
        # Product, Order, General go to END (continue chatting)
        graph.add_edge("product_agent", END)
        graph.add_edge("order_agent", END)
        graph.add_edge("general_agent", END)
        
        # Summary goes to END (conversation finished)
        graph.add_edge("summarize_agent", END)
        
        # Compile
        self.memory = MemorySaver()
        self.graph = graph.compile(checkpointer=self.memory)
    
    def route_query(self, state: ChatBotState):
        """Route the user query to the appropriate agent"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Use LLM to classify intent
        routing_messages = [
            SystemMessage(content=self.router_prompt),
            HumanMessage(content=last_message)
        ]
        
        result = self.model.invoke(routing_messages)
        intent = result.content.strip().lower()
        
        if self.debug:
            print(f"🔀 Router detected intent: {intent}")
        
        # Validate intent
        if intent not in ["product", "order", "general", "done"]:
            intent = "general"
        
        # Set is_done flag if user is ending conversation
        is_done = (intent == "done")
        
        return {"current_agent": intent, "is_done": is_done}
    
    def get_next_node(self, state: ChatBotState) -> str:
        """Return the next node based on routing"""
        return state.get("current_agent", "general")
    
    def handle_general(self, state: ChatBotState, config: dict):
        """Handle general/greeting queries"""
        if self.debug:
            print("💬 Handling general query...")
        
        messages = [SystemMessage(content=self.general_prompt)] + state["messages"]
        result = self.model.invoke(messages)
        
        return {"messages": [result], "current_agent": "general"}

    def chat(self, message: str, thread_id: str = None):
        """Send a message and get a response"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)], 
                "current_agent": "",
                "is_done": False
            },
            config=config
        )
        
        response = result["messages"][-1].content
        is_done = result.get("is_done", False)
        
        return response, thread_id, is_done


#-----------------------------------------------------------------------------
# Create the orchestrator
chatbot = ChatBotOrchestrator(model=llm, debug=True)


def main():
    """Interactive chat loop"""
    print("=" * 60)
    print("🤖 Welcome to the Laptop Store Chatbot!")
    print("=" * 60)
    print("I can help you with:")
    print("  • Product information (prices, features)")
    print("  • Order management (check orders, update quantities)")
    print("\nJust say 'goodbye' or 'I'm done' to get a summary and exit")
    print("=" * 60)
    
    thread_id = str(uuid.uuid4())
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            response, thread_id, is_done = chatbot.chat(user_input, thread_id)
            
            if is_done:
                # Summary was generated by the graph, display it and exit
                print(f"\n📋 Conversation Summary:\n{response}")
                print("\n👋 Goodbye! Thank you for using our chatbot.")
                break
            else:
                print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
