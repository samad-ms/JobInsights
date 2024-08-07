import re
import pandas as pd
# from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_valid_description(df, total_desc_count, model):
    pattern = r'\b\d+\b'  # Compile the regular expression pattern outside the loop
    for i in range(total_desc_count, -1, -1):
        desc_string = generate_description_string(df, i, False)
        total_tokens = str(model.count_tokens(desc_string))
        total_tokens_int = extract_integer_from_string(total_tokens)
        
        if total_tokens_int and total_tokens_int <= 27000: #32000 can accept by gemini
            return desc_string
    return None 

# extracting job discription from df
def generate_description_string(df, slice_number, full=False):
    if not full:
        return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'][:slice_number], start=0))
    else:
        return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'], start=0))

#eg:- this fun is used to extract 27770 "there are 27770 tokes"
def extract_integer_from_string(input_string):
    # Define the pattern to match integer numbers
    pattern = r'\b\d+\b'  # This pattern matches one or more digits (\d+) surrounded by word boundaries (\b)

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, return the integer value
    if match:
        return int(match.group())
    else:
        return None  # Return None if no integer is found in the string
    
def format_job_descriptions(df):
    formatted_output = ""
    for idx, row in df.iterrows():
        formatted_output += f"{idx + 1}. {row['title']}: {row['description']}\n"
    
    return formatted_output


def remove_unnecessary_info_from_job_description(search_term, df):
    # Combine search term with job titles for TF-IDF vectorization
    titles = df['title'].tolist()
    titles.append(search_term)
    
    # Vectorize the titles using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(titles)
    
    # Calculate cosine similarity between search term and job titles
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    
    # Get the indices of the most similar job titles
    similar_indices = cosine_similarities.argsort()[::-1]  # Sort in descending order of similarity
    
    # Filter job titles with high similarity (adjust the threshold as needed)
    relevant_indices = [idx for idx in similar_indices if cosine_similarities[idx] > 0.2]
    
    if len(relevant_indices) == 0:
        # Choose top 2 job titles if no similar titles found
        relevant_indices = similar_indices[:2].tolist()
    
    # Combine the descriptions of the relevant job titles
    formatted_output = ""
    for idx in relevant_indices:
        row = df.iloc[idx]
        formatted_output += f"{idx+1}. {row['title']} {row['description']}\n"

    return formatted_output, relevant_indices

    

