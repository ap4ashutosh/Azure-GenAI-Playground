import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter, AzureMLEndpointApiType
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

# Fetch values from environment variables
subscription_id = os.getenv("subscription_id")
resource_group = os.getenv("resource_group")
workspace = os.getenv("workspace")
endpoint_url = os.getenv("endpoint_url")
endpoint_api_key = os.getenv("endpoint_api_key")

# Create the MLClient
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace
)

def llama_chatbot():
    """
    This function initializes and returns an AzureMLChatOnlineEndpoint object for the Llama chatbot.
    """
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=endpoint_url,
        deployment_name="Meta-Llama-3-70B-Instruct-vkrgw", 
        endpoint_api_type=AzureMLEndpointApiType.serverless, 
        endpoint_api_key=endpoint_api_key,
        content_formatter=CustomOpenAIChatContentFormatter(),
        model_kwargs={"temperature": 0.45, "max_tokens": 2048, "top_p": 0.95}
    )
    return llm

def prepare_docs(pdf_docs):
    """
    Extracts content and metadata from the provided PDF documents.
    """
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    # print("Content and metadata are extracted from the documents")
    return content, metadata

def get_text_chunks(content, metadata):
    """
    Splits the provided content into text chunks using a RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=100,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

def ingest_into_vectordb(split_docs):
    """
    Ingests the given split documents into a vector database using the FAISS algorithm.
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

template = """
[INST]
-As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
-- Answer the question based on the provided documents.
-- Be direct and factual, limited to 50 words and 2-3 sentences.
-- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
-- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
-- Avoid using confirmatory phrases or any validation in your responses.
-- Do not fabricate information or include questions in your responses.
-{question}
-[/INST]
"""


def get_conversation_chain(vectordb):
    """
    Creates a conversation chain for a conversational AI system.
    """
    llama_llm = llama_chatbot()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llama_llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True
    )
    # print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain

# print(llama_chatbot().invoke("What is the capital of France?"))