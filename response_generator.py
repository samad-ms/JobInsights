import json
import streamlit as st
import google.generativeai as genai
import re
import json
# from json_retrieval_prompt import retrieve_prompt
import os

genai.configure(api_key = st.secrets['API_KEY'])
model = genai.GenerativeModel('gemini-1.0-pro')
chat = model.start_chat(history = [])

def chat_with_gemini(prompt):
    try:
        response = chat.send_message(prompt, stream = True)
        for chunk in response:
            yield chunk.text

        # Remove some chats from the history to reduce input overload to the model.
        print(model.count_tokens(chat.history))
        if len(chat.history) > 8:
            indices_to_delete = list(range(1, len(chat.history) - 3))
            for index in sorted(indices_to_delete, reverse=True):
                del chat.history[index] 

    except Exception as e:
        return st.error(f"An Error Occured! Please Try Again. {e}")


# def generate_description_string(df, slice_number, full=False):
#     if not full:
#         return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'][:slice_number], start=0))
#     else:
#         return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'], start=0))

def set_initial_message(string):
    # Clearning the chat history when user creates new dataset
    chat.history.clear()
    #print(st.session_state.desc_string)
    chat.send_message(string)


# def get_json():

#     try:
#         response = chat.send_message(retrieve_prompt(st.session_state.search_term))
#         return response.text
#     except:
#         st.error("An Error Occured! Please Try Again.")
    
# def preprocess_json_string(json_string):
#     """Returns: Cleaned Python Dictionary"""
#     inner_json_match = re.search(r'{(.+)}', json_string, re.DOTALL)
#     if inner_json_match:
#         inner_json = '{' + inner_json_match.group(1) + '}'
#         json_to_dict = json.loads(inner_json)
#         return json_to_dict
#     else:
#         st.error("Parse Error! Try Again!")

# def get_title_counts(df):
#     job_counts = df['title'].value_counts()
#     labels = job_counts.index
#     sorted_pairs = sorted(zip(job_counts, labels), reverse = True)
#     sorted_counts, sorted_labels = zip(*sorted_pairs)
#     return list(sorted_counts), list(sorted_labels)

