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

product_pricing_df = pd.read_csv(os.path.join(DATABASE_DIR, "laptop_pricing.csv"))
print(product_pricing_df)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT
)

@tool
def get_laptop_price(laptop_name:str) -> int :
    """
    This function returns the price of a laptop, given its name as input.
    It performs a substring match between the input name and the laptop name.
    If a match is found, it returns the pricxe of the laptop.
    If there is NO match found, it returns -1
    """

    #Filter Dataframe for matching names
    match_records_df = product_pricing_df[
                        product_pricing_df["Name"].str.contains(
                                                "^" + laptop_name, case=False)
                        ]
    #Check if a record was found, if not return -1
    if len(match_records_df) == 0 :
        return -1
    else:
        return match_records_df["Price"].iloc[0]




# Load, chunk and index the contents of the product features document.
loader=PyPDFLoader(os.path.join(VECTORDB_DIR, "Laptop product descriptions.pdf"))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
splits = text_splitter.split_documents(docs)

#Create a vector store with Chroma
prod_feature_store = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model
)

get_product_features = create_retriever_tool(
    prod_feature_store.as_retriever(search_kwargs={"k": 1}),
    name="Get_Product_Features",
    description="""
    This store contains details about Laptops. It lists the available laptops
    and their features including CPU, memory, storage, design and advantages
    """
)