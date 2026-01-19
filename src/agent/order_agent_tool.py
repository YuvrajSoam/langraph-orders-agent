from langchain_core.tools import tool
import pandas as pd
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "..", "Database")
VECTORDB_DIR = os.path.join(BASE_DIR, "..", "Vectordb")

orders_df = pd.read_csv(os.path.join(DATABASE_DIR, "laptop_orders.csv"))
print(orders_df)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT
)

@tool
def get_order_details(order_id: str) -> str:
    """
    This function returns the details of an order, given its order ID as input.
    It performs a substring match between the input order ID and the order ID.
    If a match is found, it returns the details of the order.
    If there is NO match found, it returns -1
    """

        #Filter Dataframe for matching names
    match_records_df = orders_df[
                        orders_df["Order ID"] == order_id
                        ]
    #Check if a record was found, if not return -1
    if len(match_records_df) == 0 :
        return -1
    else:
        return match_records_df.iloc[0].to_dict()

@tool
def update_order_quantity(order_id: str, quantity: int) -> str:
    """
    This function updates the quantity of an order, given its order ID and the new quantity as input.
    It performs a substring match between the input order ID and the order ID.
    If a match is found, it updates the quantity of the order.
    If there is NO match found, it returns -1
    """
    match_records_df = orders_df[
                        orders_df["Order ID"] == order_id
                        ]
    if len(match_records_df) == 0 :
        return -1
    else:
        orders_df.loc[orders_df["Order ID"] == order_id, "Quantity Ordered"] = quantity
        orders_df.to_csv(os.path.join(DATABASE_DIR, "laptop_orders.csv"), index=False)
        return True

    return "Order details"