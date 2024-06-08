from typing import Any, List, Tuple
from pypdf import PdfReader
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY']=st.secrets['PINECONE_API_KEY']
pinecone_index_name= st.secrets['PINECONE_INDEX_NAME']




#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(embeddings,docs,pinecone_index_name=pinecone_index_name):
    vectordb = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return vectordb

def similar_docs(vectordb: object, query: str, k: int, unique_id: Any) -> List[Tuple[Any, float]]:
    similar_docs = vectordb.similarity_search_with_score(query,int(k),{"unique_id":unique_id})
    return similar_docs

# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke([current_doc])
    return summary

#----------------------------summarization----------------------------------


def select_frequent_keywords(large_corpus, num_keywords=1000):
    # Tokenize the corpus into words
    tokens = word_tokenize(large_corpus.lower())

    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Calculate word frequencies
    word_freq = Counter(tokens)

    # Select the most frequent keywords
    most_common_words = word_freq.most_common(num_keywords)
    common_keywords_corpus = ", ".join([word for word, freq in most_common_words])

    return common_keywords_corpus

def create_job_descriptiion_demo(job_description,search_term):
    common_keywords_corpus=select_frequent_keywords(job_description)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    template = """
    Write the entire job description in proper structure for {search_term} role.
    by considering the most frequent key words : {common_keywords_corpus}
    """
    
    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["search_term", "common_keywords_corpus"],
        template=template,
    )
    response = llm.invoke(prompt.format(search_term=search_term, common_keywords_corpus=common_keywords_corpus))
    return response.content