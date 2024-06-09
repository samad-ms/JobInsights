
from StringLineLoader import StringLineLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_TRACING_V2']=st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

def jd_to_vectorestore(data):
    loader=StringLineLoader(data)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents,OpenAIEmbeddings())
    return db

def get_context_retriever(vectore_store):
    # Define the prompt template for the retriever
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', "{input}"),
        ('user', "Given the above conversation, generate a search query to look up in order to get the relevant information based on the conversation")
    ])
    llm = ChatOpenAI()
    retriever = vectore_store.as_retriever()
    
    # Create a history-aware retriever chain
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, retriever, prompt
    )
    return history_aware_retriever_chain

def get_rag_chain(history_aware_retriever_chain):
    # Define the prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever_chain, stuff_documents_chain)
    return rag_chain

def get_response(db,user_query,history):
    context_retriever_chain = get_context_retriever(vectore_store=db)
    rag_chain = get_rag_chain(context_retriever_chain)
    
    response = rag_chain.invoke({
        "chat_history": history,
        "input": user_query
    })
    return response['answer']
