import streamlit as st 
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import json
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
from langchain_core.prompts import PromptTemplate

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


# def decode_json_to_natural_language(data):
#     template = """Convert the following data to a natural language description suitable for job seekers:

#     {json_data}

#     Provide key insights and trends from the dataset that job seekers can use to improve their job search. Additionally, suggest areas for learning that can potentially lead to career growth and a salary hike based on the insights from the data."""

#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
#     prompt = PromptTemplate(
#         input_variables=["json_data"],
#         template=template,
#     )
#     response = llm.invoke(prompt.format(json_data=data))
#     return response.content


def auto_summarizer():
    lida = Manager(text_gen = llm("openai"))
    textgen_config = TextGenerationConfig(n=1, temperature=0.1, model="gpt-3.5-turbo-0301", use_cache=True)
    summary = lida.summarize("filename.csv",summary_method="default", textgen_config=textgen_config)
    # insights=decode_json_to_natural_language(summary)
    goals = lida.goals(summary, n=3, textgen_config=textgen_config)

    charts = lida.visualize(summary=summary, goal=goals[0], textgen_config=textgen_config, library="seaborn")  
    img_base64_string = charts[0].raster
    img = base64_to_image(img_base64_string)

    charts2 = lida.visualize(summary=summary, goal=goals[1], textgen_config=textgen_config, library="seaborn")  
    img_base64_string2 = charts2[0].raster
    img2 = base64_to_image(img_base64_string2)

    return goals,img,img2
    
