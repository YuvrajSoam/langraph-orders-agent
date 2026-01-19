from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# from IPython.display import Image
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
import uuid
import json
from order_agent_tool import get_order_details, update_order_quantity
from dotenv import load_dotenv
import os
load_dotenv()
from IPython.display import Image

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


#An Agent State class that keep state of the agent while it answers a query
class OrderAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

#-----------------------------------------------------------------------------
#An agent class that manages all agentic interactions
class OrderAgent:

    #Setup the agent graph, tools and memory
    def __init__(self, model, tools, system_prompt, checkpoint_saver, debug):

        #attach tools to model
        self.model=model.bind_tools(tools)
        self.system_prompt=system_prompt
        self.debug=debug
        tool_node = ToolNode(tools)


        #Setup the graph for the agent manually
        agent_graph=StateGraph(OrderAgentState)
        agent_graph.add_node("order_llm",self.call_llm)
        agent_graph.add_node("tools",tool_node)
        agent_graph.add_conditional_edges(
            "order_llm",
            tools_condition,
        )
        agent_graph.add_edge("tools","order_llm")
        #Set where there graph starts
        agent_graph.set_entry_point("order_llm")

        #Add chat memory
        #compile the graph
        self.agent_graph = agent_graph.compile(checkpointer=checkpoint_saver)

        #Setup tools
        self.tools = { tool.name : tool for tool in tools }
        if self.debug:
            print("\nTools loaded :", self.tools)
            



    #Call the LLM with the messages to get next action/result
    def call_llm(self, state:OrderAgentState):
        
        messages=state["messages"]

        #If system prompt exists, add to messages in the front
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
            
        #invoke the model with the message history
        result = self.model.invoke(messages)
        if self.debug:
            print(f"\nLLM Returned : {result}")
        #Return the LLM output
        return { "messages":[result] }
    

#-----------------------------------------------------------------------------
#Setup the custom agent

#Note that this is a string, since the model init only accepts a string.
system_prompt = """
    You are professional chatbot that manages orders for laptops sold by your company.
    To manage orders, you will ONLY use the available tools and NOT your own memory.
    The tools allow for retrieving order details as well as update order quantity.
    Do NOT reveal information about other orders than the one requested.
    You will handle small talk and greetings by producing professional responses.
    """

checkpoint_saver=MemorySaver()


#Create the custom product agent
order_agent = OrderAgent(llm, 
                           [get_order_details, update_order_quantity], 
                           system_prompt,
                           checkpoint_saver,
                           debug=False)


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    Image(order_agent.agent_graph.get_graph().draw_mermaid_png())
    # result = order_agent.agent_graph.invoke(
    #     {"messages": [HumanMessage(content="What is the quantity ordered for order ID 1234567890?")]},
    #     config=config
    # )
    # print(result["messages"][-1].content)

if __name__ == "__main__":
    main()


        