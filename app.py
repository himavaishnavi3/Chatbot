import streamlit as st
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

st.title("🤖 RAG Chatbot")
st.write("Ask anything about company policies")

# Load documents
documents = []
folder_path = "Rag documents"   

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    if file.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Store in FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# User input
query = st.text_input("Ask your question:")

if query:
    retrieved_docs = retriever.get_relevant_documents(query)

    if len(retrieved_docs) > 0:
        context = retrieved_docs[0].page_content

        sentences = context.split(".")
        answer = ""

        for sentence in sentences:
            if any(word in sentence.lower() for word in query.lower().split()):
                answer = sentence.strip()
                break

        if answer == "":
            answer = context

        st.subheader("Answer:")
        st.write(answer)
    else:
        st.write("No relevant answer found.")