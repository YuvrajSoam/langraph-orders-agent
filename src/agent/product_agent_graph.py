from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# from IPython.display import Image
from langchain_openai import AzureChatOpenAI
from tools import get_laptop_price, get_product_features
from langgraph.prebuilt import ToolNode, tools_condition
import uuid
import json
from dotenv import load_dotenv
import os
load_dotenv()

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
class ProductAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

#-----------------------------------------------------------------------------
#An agent class that manages all agentic interactions
class ProductAgent:

    #Setup the agent graph, tools and memory
    def __init__(self, model, tools, system_prompt, checkpoint_saver, debug):

        #attach tools to model
        self.model=model.bind_tools(tools)
        self.system_prompt=system_prompt
        self.debug=debug
        tool_node = ToolNode(tools)


        #Setup the graph for the agent manually
        agent_graph=StateGraph(ProductAgentState)
        agent_graph.add_node("product_llm",self.call_llm)
        agent_graph.add_node("tools",tool_node)
        agent_graph.add_conditional_edges(
            "product_llm",
            tools_condition,
        )
        agent_graph.add_edge("tools","product_llm")
        #Set where there graph starts
        agent_graph.set_entry_point("product_llm")

        #Add chat memory
        #compile the graph
        self.agent_graph = agent_graph.compile(checkpointer=checkpoint_saver)

        #Setup tools
        self.tools = { tool.name : tool for tool in tools }
        if self.debug:
            print("\nTools loaded :", self.tools)
            



    #Call the LLM with the messages to get next action/result
    def call_llm(self, state:ProductAgentState):
        
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
    You are professional chatbot that answers questions about laptops sold by your company.
    To answer questions about laptops, you will ONLY use the available tools and NOT your own memory.
    You will handle small talk and greetings by producing professional responses.
    """

checkpoint_saver=MemorySaver()


#Create the custom product agent
product_agent = ProductAgent(llm, 
                           [get_laptop_price, get_product_features], 
                           system_prompt,
                           checkpoint_saver,
                           debug=False)


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = product_agent.agent_graph.invoke(
        {"messages": [HumanMessage(content="What is the price of the laptop called 'AlphaBook Pro'?")]},
        config=config
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()

#Visualize the Agent
# Image(product_agent.agent_graph.get_graph().draw_mermaid_png())
        