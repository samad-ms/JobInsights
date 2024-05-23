import os

def generate_streamlit_folder():
    # Create .streamlit folder if it doesn't exist
    streamlit_folder = ".streamlit"
    if not os.path.exists(streamlit_folder):
        os.makedirs(streamlit_folder)

    # Check if secrets.toml already exists
    secrets_file = os.path.join(streamlit_folder, "secrets.toml")
    if os.path.exists(secrets_file):
        print("Secrets file already exists.")
        return

    # Ask user for API key
    api_key = input("Enter your API key: ").strip()

    # Write API key to secrets.toml
    with open(secrets_file, "w") as f:
        f.write(f"API_KEY=\"{api_key}\"\n")

    print("API key has been saved to secrets.toml.")

# Call the function to generate the .streamlit folder and save API key
generate_streamlit_folder()
