import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = "7b42e3b6f2ef4a83a7f898fa0e62ed8f" 
OPENAI_DEPLOYMENT_ENDPOINT = "https://dg1-sn.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "DG1-SN-GPT-35-TURBO"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_DEPLOYMENT_VERSION = "2024-03-01"

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "DG1-SN-TEXT-EMBEDDING-ADA-002"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"



def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])


def ask_question_with_context(qa, question, chat_history):
    query = "what is Azure OpenAI Service?"
    result = qa.invoke({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


if __name__ == "__main__":
    # Configure OpenAI API
    openai.api_type = "azure"
    openai.api_version = OPENAI_DEPLOYMENT_VERSION
    openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
    openai.api_key = OPENAI_API_KEY
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_type="azure")
    
    embeddings=AzureOpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_key=OPENAI_API_KEY,
                                openai_api_type="azure",
                                chunk_size=1)


    # Initialize gpt-35-turbo and our embedding model
    #load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    #use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)


    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)