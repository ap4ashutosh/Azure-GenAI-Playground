# This is the backend file for the chat application using models from Azure ML model catalog
# Written by Ashutosh 
# Date: 15th june 2024

import os
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter, AzureMLEndpointApiType
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Fetch values from environment variables
subscription_id = os.getenv("subscription_id")
resource_group = os.getenv("resource_group")
workspace = os.getenv("workspace")
url = os.getenv("endpoint_url")
api = os.getenv("endpoint_api_key")


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
    llm = AzureMLChatOnlineEndpoint(endpoint_url=url, 
                                 deployment_name="Meta-Llama-3-8B-Instruct-proj", 
                                 endpoint_api_type=AzureMLEndpointApiType.serverless, 
                                 endpoint_api_key=api,
                                 content_formatter=CustomOpenAIChatContentFormatter())
    return llm

def chatbot_memory():
    llm = llama_chatbot()
    memory = ConversationBufferMemory(llm = llm, 
                                      max_token_limit = 2048)
    return memory

def chatbot_conversation(input_txt:str, memory):
    """
    Initializes and returns a conversation chain for a chatbot.

    Returns:
        ConversationChain: A conversation chain object with the specified LLM and memory.
    """
    llm = llama_chatbot()
    conversation = ConversationChain(llm = llm, 
                                     memory = memory,
                                     output_key = "text")
    reply = conversation.predict(input = input_txt)
    return reply

# print(llama_chatbot().invoke("What is the capital of India?"))