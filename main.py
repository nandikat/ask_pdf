import streamlit as st
from dotenv import load_dotenv
from ask_pdf import process_pdf,rag
load_dotenv()

st.set_page_config(page_title="Ask your PDfs", page_icon="📄")
st.title("📄Ask your PDF questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing document..."):
                try:
                    retriever = process_pdf(uploaded_file)
                    st.session_state.rag = rag(retriever)                    
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success("Document successfully indexed!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    
if "rag" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask questions from your PDF"):
        
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Don't  have time to go through the entire document? Upload your PDF and ask questions from it!")
