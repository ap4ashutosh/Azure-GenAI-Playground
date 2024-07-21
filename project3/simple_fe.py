import streamlit as st
import conv_be as bot
from langchain_core.messages import HumanMessage
import PyPDF2
import os

st.set_page_config(page_title="Chatbot", page_icon=":robot:")

st.title("Hello users 🙋‍♂️ this is the RAG chatbot from project Azure-GenAI-playground")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF(s) 📚")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            pdf_paths = []
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(f"temp_{uploaded_file.name}")
            
            # Process PDFs
            content, metadata = bot.prepare_docs(pdf_paths)
            split_docs = bot.get_text_chunks(content, metadata)
            vectordb = bot.ingest_into_vectordb(split_docs)
            
            # Create conversation chain
            st.session_state.conversation_chain = bot.get_conversation_chain(vectordb)
            
            # Clean up temporary files
            for path in pdf_paths:
                os.remove(path)
            
        st.success(f"Processed {len(uploaded_files)} PDF(s)")

# Main chat interface
if 'conversation_chain' not in st.session_state:
    st.warning("Please upload and process PDFs to start the conversation.")
else:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

    input_text = st.chat_input("Chat with RAG llama : 🦙")
    if input_text:
        with st.chat_message('user'):
            st.markdown(input_text)

        st.session_state.chat_history.append({'role':'user','text':input_text})
        
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain({"question": input_text})
            chat_response = response['answer']
        
        with st.chat_message('assistant'):
            st.markdown(chat_response)

        st.session_state.chat_history.append({'role':'assistant', 'text':chat_response})