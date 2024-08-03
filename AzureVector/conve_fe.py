import streamlit as st
import conv_be as bot
from langchain_core.messages import HumanMessage
import PyPDF2
import os

st.set_page_config(page_title="RAG Llama Chatbot", page_icon="🦙", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .sidebar-content {
        padding: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🦙 RAG Llama Chatbot")
    st.subheader("Your AI-powered PDF assistant")

with col2:
    st.image("https://img.icons8.com/color/96/000000/llama.png", width=100)

# Sidebar
with st.sidebar:
    st.header("📚 Document Upload")
    with st.expander("Upload and Process PDFs", expanded=True):
        uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        
        if uploaded_files:
            st.write(f"📂 {len(uploaded_files)} file(s) uploaded")
            if st.button("🚀 Process PDFs", key="process_pdfs"):
                with st.spinner("🔮 Magic in progress..."):
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        with open(f"temp_{uploaded_file.name}", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(f"temp_{uploaded_file.name}")
                    
                    content, metadata = bot.prepare_docs(pdf_paths)
                    split_docs = bot.get_text_chunks(content, metadata)
                    vectordb = bot.ingest_into_vectordb(split_docs)
                    
                    st.session_state.conversation_chain = bot.get_conversation_chain(vectordb)
                    
                    for path in pdf_paths:
                        os.remove(path)
                
                st.success(f"🎉 Successfully processed {len(uploaded_files)} PDF(s)!")
                st.balloons()

# Main chat interface
if 'conversation_chain' not in st.session_state:
    st.info("👆 Please upload and process PDFs in the sidebar to start chatting!")
else:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

    input_text = st.chat_input("🦙 Ask me anything about your PDFs...")
    if input_text:
        with st.chat_message('user'):
            st.markdown(input_text)

        st.session_state.chat_history.append({'role':'user','text':input_text})
        
        with st.spinner("🧠 Thinking..."):
            response = st.session_state.conversation_chain({"question": input_text})
            chat_response = response['answer']
        
        with st.chat_message('assistant'):
            st.markdown(chat_response)

        st.session_state.chat_history.append({'role':'assistant', 'text':chat_response})

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for the Azure GenAI Project")