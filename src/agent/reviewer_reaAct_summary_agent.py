from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# from IPython.display import Image
from langchain_openai import AzureChatOpenAI
import uuid
import json
from langgraph.graph import START
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


class ReviewerReadActSummaryAgentState(TypedDict):
    # State will store the conversation (as a list of AnyMessage) and also a retry count (for the review node)
    messages: Annotated[list[AnyMessage], operator.add]
    feedback: str
    retry_count: int

class ReviewerReadActSummaryAgent:
    def __init__(self, model, summary_prompt, reviewer_prompt, checkpoint_saver, debug):
        self.summary_prompt = summary_prompt
        self.model = model
        self.debug = debug
        self.reviewer_prompt = reviewer_prompt

        agent_graph = StateGraph(ReviewerReadActSummaryAgentState)
        agent_graph.add_node("generate_summary", self.generate_summary)
        agent_graph.add_node("review_summary", self.review_summary)
        agent_graph.add_edge(START, "generate_summary")
        agent_graph.add_edge("review_summary", "generate_summary")
        agent_graph.add_conditional_edges(
            "generate_summary",
            self.should_continue,
            {
                True: "review_summary",   # Continue reviewing if under retry limit
                False: END                 # Stop if exceeded retry limit
            }
        )
        agent_graph.set_entry_point("generate_summary")

        self.agent_graph = agent_graph.compile(checkpointer=checkpoint_saver)

    def generate_summary(self, state:ReviewerReadActSummaryAgentState):
        messages=state["messages"]

        #Prepend reviewer system prompt to messages
        messages = [SystemMessage(content=self.summary_prompt)] + messages
        
        #invoke the summary generator with the message history
        result = self.model.invoke(messages)
        
        if self.debug:
            print(f"==============\n Generator returned output : {result.content}")
        return { "messages":[result] }

    def review_summary(self, state:ReviewerReadActSummaryAgentState):
        messages=state["messages"]
        messages = [SystemMessage(content=self.reviewer_prompt)] + messages

        #invoke the reviewer with the message history
        feedback = self.model.invoke(messages)

        # Increment retry count after each review
        new_retry_count = state.get("retry_count", 0) + 1
        
        if self.debug:
            print(f"*************\n Reviewer returned output : {feedback.content}")
            print(f"Retry count: {new_retry_count}")
        return { "messages":[feedback], "feedback": feedback.content, "retry_count": new_retry_count }

    def should_continue(self, state:ReviewerReadActSummaryAgentState):
        # Get retry count with default of 0 if not set
        retry_count = state.get("retry_count", 0)
        
        # Stop if retry count > 3, otherwise continue to review
        if retry_count > 3:    
            return False
        else:
            return True


summarizer_prompt="""
You are an document summarizer who can summarize a document provide to you.
For the input provided, create a summary with less than 50 words.
If the user has provides critique, responsed with a revised version of your previous attempts
"""

reviewer_prompt="""
You are a reviewer grading summaries for an article. 
Compare the user input document and generated summary.
Check if the summary accurately reflects the contents of the document.
Provide recommendations for improvement in less than 50 words.
"""



checkpoint_saver=MemorySaver()


#Create the custom product agent
reviewer_reaAct_summary_agent = ReviewerReadActSummaryAgent(llm, 
                           summarizer_prompt,
                           reviewer_prompt,
                           checkpoint_saver,
                           debug=False)


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    document_to_summarize = """
    What is Lorem Ipsum?
    Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
    Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
    when an unknown printer took a galley of type and scrambled it to make a type 
    specimen book. It has survived not only five centuries, but also the leap into 
    electronic typesetting, remaining essentially unchanged.
    
    Why do we use it?
    It is a long established fact that a reader will be distracted by the readable 
    content of a page when looking at its layout. The point of using Lorem Ipsum is 
    that it has a more-or-less normal distribution of letters, making it look like 
    readable English.
    
    Where does it come from?
    Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots 
    in a piece of classical Latin literature from 45 BC, making it over 2000 years old.
    Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" 
    (The Extremes of Good and Evil) by Cicero, written in 45 BC.
    """
    
    result = reviewer_reaAct_summary_agent.agent_graph.invoke(
        {
            "messages": [HumanMessage(content=document_to_summarize)],
            "retry_count": 0,
            "feedback": ""
        },
        config=config
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()