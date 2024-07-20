from dotenv import find_dotenv, load_dotenv
import os
load_dotenv(find_dotenv())

print(os.getenv("endpoint_api_key"))