
from StringLineLoader import StringLineLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
os.environ['OPENAI_API_KEY']=st.secrets['GEMINI_API_KEY']
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_TRACING_V2']=st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

def jd_to_vectorestore(data):
    loader=StringLineLoader(data)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    db = Chroma.from_documents(documents,OpenAIEmbeddings())
    return db

def retrive_data_and_respose(db,input):
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context(context is job discriptions). 
        Think well before providing a detailed answer. 
        <context>
        {context}
        </context>
        Question: {input}""")
    llm=ChatOpenAI()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=db.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":input})
    return response['answer']
    
    
