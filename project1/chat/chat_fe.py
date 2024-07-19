# this is the file where we will use streamlit to create a frontend for our chatbot application
# will use the backend file chat_be.py

import streamlit as st
import chat_be as bot
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Chatbot", page_icon=":robot:")

st.title("Hello users this is the chatbot from GenAI-playground")

# we need to create the memory to store all the messages using streamlit session memory
if 'memory' not in st.session_state:
    st.session_state.memory = bot.chatbot_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

input_text = st.chat_input("Chat with myllama :robot_face:")
if input_text:
    with st.chat_message('user'):
        st.markdown(input_text)

    st.session_state.chat_history.append({'role':'user','text':input_text})
    
    chat_response = bot.chatbot_conversation(input_txt=input_text, memory=st.session_state.memory)
    
    with st.chat_message('assistant'):
        st.markdown(chat_response)

    st.session_state.chat_history.append({'role':'assistant', 'text':chat_response})
