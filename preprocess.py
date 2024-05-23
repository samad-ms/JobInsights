import re
# from response_generator import generate_description_string

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
    
def generate_description_string(df, slice_number, full=False):
    if not full:
        return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'][:slice_number], start=0))
    else:
        return '\n'.join('{}. {}'.format(i + 1, desc.replace("\n", " ")) for i, desc in enumerate(df['description'], start=0))


def find_valid_description(df, total_desc_count, model):
    pattern = r'\b\d+\b'  # Compile the regular expression pattern outside the loop
    for i in range(total_desc_count, -1, -1):
        desc_string = generate_description_string(df, i, False)
        total_tokens = str(model.count_tokens(desc_string))
        total_tokens_int = extract_integer_from_string(total_tokens)
        
        if total_tokens_int and total_tokens_int <= 27000: #32000 can accept by gemini
            return desc_string
    return None 
