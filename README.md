# 📄 Ask PDF: RAG-based PDF Chatbot

This is a Retrieval-Augmented Generation (RAG) application that allows users to upload any PDF document and chat with it in real-time. It answers questions based strictly on the contents of the uploaded document.

## Features

* **Secure & Local Processing:** Your PDF is parsed locally and converted into math vectors stored on your own machine.
* **Intelligent Retrieval (RAG):** Answers are grounded *only* in the uploaded document to prevent hallucinations.
* **Conversational Memory:** The chat interface retains the history of your session.
* **Streamlit UI:** A clean, easy-to-use web interface.

## Tech Stack

* **Frontend UI:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Google Gemini 2.5 Flash 
* **Embeddings:** Google Gemini Embeddings 
* **Vector Database:** ChromaDB
* **Document Parsing:** pypdf

