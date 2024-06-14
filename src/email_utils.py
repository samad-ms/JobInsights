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
def get_email_Response(email_topic, email_sender, email_recipient,email_type,Signature,candidate_details_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    # Template for building the PROMPT
    # Template for building the PROMPT
    template = """
        Write the entire email in a proper structure(but still concise), including the subject and a formal style, on the topic: {email_topic}. Also, consider including a brief candidate information (candidate information must have less imbact on this email and dont consider projects information from it ): {candidate_details_text}, if necessary.

        Type: {email_type}
        Sender: {sender}
        Recipient: {recipient}

        Email Text:

        Signature: {Signature}
    """
    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["email_topic","candidate_details_text","email_type", "sender", "recipient","Signature"],
        template=template,
    )

    # Generating the response using LLM
    # Using 'invoke' function for the response
    response = llm.invoke(prompt.format(email_topic=email_topic, sender=email_sender, recipient=email_recipient,email_type=email_type,Signature=Signature,candidate_details_text=candidate_details_text))
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
