import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated Imports for HuggingFaceEmbeddings and OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # Updated import
from langchain.prompts import PromptTemplate

# For chain creation using new constructors
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.title("RAG System with DeepSeek R1 & Ollama")

# Create an expander for debug logs
debug_logs = st.expander("Debug Logs", expanded=False)

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    debug_logs.write("Uploaded file saved as 'temp.pdf'.")
    logger.info("Uploaded file saved as 'temp.pdf'.")

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    debug_logs.write(f"Loaded {len(docs)} page(s).")
    logger.info(f"Loaded {len(docs)} page(s).")
    if docs:
        # Show a preview of the first document's content
        debug_logs.write(f"First page preview: {docs[0].page_content[:200]}")

    # Initialize the HuggingFaceEmbeddings with explicit model_name
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    debug_logs.write("Initialized HuggingFaceEmbeddings for text splitting.")
    logger.info("Initialized HuggingFaceEmbeddings for text splitting.")

    text_splitter = SemanticChunker(hf_embeddings)
    documents = text_splitter.split_documents(docs)
    debug_logs.write(f"Documents split into {len(documents)} chunk(s).")
    logger.info(f"Documents split into {len(documents)} chunk(s).")
    if documents:
        debug_logs.write(f"First chunk preview: {documents[0].page_content[:200]}")

    # Create vector store with explicit embedding
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    debug_logs.write(f"Vector store created and retriever configured {retriever}.")
    logger.info("Vector store created and retriever configured {retriever}.")

    # Updated Ollama LLM
    llm = OllamaLLM(model="deepseek-r1:1.5b")
    debug_logs.write(f"Initialized OllamaLLM {llm}.")
    logger.info(f"Initialized OllamaLLM {llm}.")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {input}
    Answer:"""
    QA_PROMPT = PromptTemplate.from_template(prompt)
    debug_logs.write(f"Prompt template created {QA_PROMPT}.")
    logger.info(f"Prompt template created {QA_PROMPT}.")

    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT, document_variable_name="context")
    qa = create_retrieval_chain(combine_docs_chain=combine_documents_chain, retriever=retriever)
    debug_logs.write(f"Retrieval chain created {qa}.")
    logger.info(f"Retrieval chain created {qa}.")

    user_input = st.text_input("Ask a question about your document:")

    if user_input:

        debug_logs.write(f"User input received: {user_input}")
        logger.info(f"User input received: {user_input}")

        result = qa.invoke({"input": user_input})
        debug_logs.write(f"Chain output: {result}")
        logger.info(f"Chain output: {result}")

        st.write("Response:")
        st.write(result["answer"])