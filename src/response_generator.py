import json
import streamlit as st
import google.generativeai as genai
import re
import json
import os

genai.configure(api_key = st.secrets['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.0-pro')
chat = model.start_chat(history = [])

def chat_with_gemini(prompt):
    try:
        # response = chat.send_message(prompt, stream = True)
        # for chunk in response:
        #     yield chunk.text
        response = chat.send_message(prompt)
        # return response["candidates"][0]["content"]["parts"][0]["text"]
        return response.candidates[0].content.parts[0].text

        # Remove some chats from the history to reduce input overload to the model.
        print(model.count_tokens(chat.history))
        if len(chat.history) > 8:
            indices_to_delete = list(range(1, len(chat.history) - 3))
            for index in sorted(indices_to_delete, reverse=True):
                del chat.history[index] 

    except Exception as e:
        return st.error(f"An Error Occured! Please Try Again. {e}")

def set_initial_message(string):
    # Clearning the chat history when user creates new dataset
    chat.history.clear()
    #print(st.session_state.desc_string)
    chat.send_message(string)

