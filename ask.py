import os
from flask import Flask,request,jsonify
from dotenv import load_dotenv
from google import genai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
def ask(pdf_path: str, user_question: str):
    
    print("Loading and parsing PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("Generating embeddings and building vector database...")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview" 
    )

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("Formulating answer using Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2 
    )

    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_question)

if __name__ == "__main__":
    my_pdf = "report.pdf" 
    query = "What are the key takeaways?"
    
    print(ask(my_pdf, query))