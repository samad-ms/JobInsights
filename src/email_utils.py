import streamlit as st 
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import PyPDF2
from dotenv import load_dotenv
load_dotenv()

# Function to get the response back
def get_email_Response(email_topic, email_sender, email_recipient,Signature,candidate_details_text,company_details_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    # Template for building the PROMPT
    # Template for building the PROMPT
    template = """
        Write the entire email in a proper structure(but still concise), including the subject and a formal style, on the topic: {email_topic}. Also, consider including a 'few lines of candidate information' (candidate information must have less imbact on this email information from it ): {candidate_details_text}, if necessary.
        also include company information and why I am intrested in company :{company_details_text}

        Sender: {sender}
        Recipient: {recipient}

        Email Text:

        Signature: {Signature}

    


    """
    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["email_topic","candidate_details_text","company_details_text","sender", "recipient","Signature"],
        template=template,
    )

    # Generating the response using LLM
    # Using 'invoke' function for the response
    response = llm.invoke(prompt.format(email_topic=email_topic,candidate_details_text=candidate_details_text,company_details_text=company_details_text, sender=email_sender, recipient=email_recipient,Signature=Signature))
    return response

#------------function to fetch data from website---------------------------------------------------------------------------


# Function to create a vector store from a website URL
def get_vectore_store_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents=document)
    vectore_store = FAISS.from_documents(documents=document_chunks, embedding=OpenAIEmbeddings())
    return vectore_store

def get_rag_chain(vector_store):
    prompt = ChatPromptTemplate.from_template("""
        Based on the provided context, retrieve and compile detailed information about the candidate. 
        The information should include:
        - Professional Summary
        - Experience
        - Education
        - Certifications
        - Any other relevant details

        <context>
        {context}
        </context>
        Question: {input}
    """)

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


def get_candidate_details(website_url,candidate_name):
    vectore_store=get_vectore_store_url(url=website_url)
    rag_chain = get_rag_chain(vectore_store)
    # Get the response from the RAG chain
    response = rag_chain.invoke({
        "input": f"Please provide detailed information about the candidate {candidate_name} , including their professional summary, experience, education, and certifications."
    })
    return response['answer']

#-----------------scraping company info -------------------


def get_company_details(website_url):
    company_vectore_store=get_comapny_vectore_store_url(url=website_url)
    company_rag_chain = get_company_rag_chain(company_vectore_store)
    # Get the response from the RAG chain
    response = company_rag_chain.invoke({
        "input": f"Please provide few important information about the company , including a concise company overview, mission and vision, key products/services, recent achievements/news, and company culture."
    })
    return response['answer']




def get_comapny_vectore_store_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents=document)
    vectore_store = FAISS.from_documents(documents=document_chunks, embedding=OpenAIEmbeddings())
    return vectore_store

def get_company_rag_chain(vector_store):
    prompt = ChatPromptTemplate.from_template("""
        Based on the provided context, retrieve and compile concise information about the company. 
        The information should include a few lines about:
        - Company Overview
        - Mission and Vision
        - Key Products/Services
        - Recent Achievements/News
        - Company Culture

        <context>
        {context}
        </context>
        Question: {input}
    """)

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain
