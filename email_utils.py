import streamlit as st 
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI   
from dotenv import load_dotenv
load_dotenv()

# Function to get the response back
def get_email_Response(form_input, email_sender, email_recipient,email_type, email_style,Signature):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    # Template for building the PROMPT
    template = """
    Write the entire email in proper structure (including the subject...) with formal style on the topic: {email_topic}.
    Type: {email_type}
    Sender: {sender}
    Recipient: {recipient}
    Email Text:
    Signature: {Signature}
    """
    
    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["style", "email_topic","email_type", "sender", "recipient","Signature"],
        template=template,
    )

    # Generating the response using LLM
    # Using 'invoke' function for the response
    response = llm.invoke(prompt.format(email_topic=form_input, sender=email_sender, recipient=email_recipient,email_type=email_type, style=email_style,Signature=Signature))
    return response
