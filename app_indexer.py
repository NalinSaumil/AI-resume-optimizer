from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai

OPENAI_API_KEY = "e135ffd6b14f4957a3b94ac1c8ba91d4" 
OPENAI_DEPLOYMENT_ENDPOINT = "https://dg1-sn.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "DG1-SN-GPT-35-TURBO"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_DEPLOYMENT_VERSION = "2024-03-01"

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "DG1-SN-TEXT-EMBEDDING-ADA-002"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

if __name__ == "__main__":
    embeddings=AzureOpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_key=OPENAI_API_KEY,
                                openai_api_type="azure",
                                chunk_size=1)
    dataPath = "./data/documentation/"
    fileName = dataPath + "resume.pdf"

    #use langchain PDF loader
    loader = PyPDFLoader(fileName)

    #split the document into chunks
    pages = loader.load_and_split()

    #Use Langchain to create the embeddings using text-embedding-ada-002
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    #save the embeddings into FAISS vector store
    db.save_local("./dbs/documentation/faiss_index")