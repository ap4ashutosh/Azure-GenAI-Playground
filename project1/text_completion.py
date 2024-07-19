import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
 
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter, AzureMLEndpointApiType
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv(dotenv_path="project1\chat\.env")

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

from langchain_core.messages import HumanMessage

llm = AzureMLChatOnlineEndpoint(endpoint_url=url, 
                                 deployment_name="Meta-Llama-3-8B-Instruct-mybot", 
                                 endpoint_api_type=AzureMLEndpointApiType.serverless, 
                                 endpoint_api_key=api,
                                 content_formatter=CustomOpenAIChatContentFormatter())


input_text = input("Enter a text needs to be completed: ")
response = llm.invoke([HumanMessage(input_text)])
print(response)

# print(f"URL: {url}")
# print(f"API Key: {api}")  # Be cautious about printing API keys in production