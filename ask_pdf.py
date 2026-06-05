from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def process_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    documents = []  
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    # vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_model,
        persist_directory="./chroma_db" 
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 3})


def rag(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)   
    template = """
    You are a helpful assistant specialized in document analysis. 
    Use the following pieces of retrieved context to answer the question accurately.
    If the answer cannot be found in the context, say exactly: "I cannot find the answer in the provided document."
    Do not make up facts.    
    Question: {question}     
    Context: {context}     
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}" for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain