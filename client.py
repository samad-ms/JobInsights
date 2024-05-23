import streamlit as st

#----------------------------------------------------------------------------------------
def home_tab():
    """ Home Tab """
    st.title("🔍 Welcome to JobInsights!")
    st.write("""
        #### What is it?

        **JobInsights** is an AI-powered, advanced job searching and interactive application designed to streamline your job search process. It gathers data from popular websites such as Indeed, LinkedIn, Glassdoor, and ZipRecruiter, providing a personalized and comprehensive job summary and insights for each query.

        #### How it Works?

        **JobInsights** leverages an advanced Large Language Model (LLM), GPT-3.5-turbo by OpenAI, to swiftly analyze and comprehend hundreds of job descriptions. This AI extracts vital details including required skills, experience levels, interview insights, and even offers tutorials and guides for job preparation. This automated process saves significant time by eliminating the need for manual sorting through numerous job postings.

        #### Features

        * **Job Summarization:** Automatically generates concise summaries of job descriptions, highlighting key details for quick understanding of job requirements.
        * **Skill Matching:** Identifies the most relevant skills for each job, allowing you to tailor your resume and cover letter effectively.
        * **Interview Insights:** Provides interview questions and tips based on job descriptions, aiding in interview preparation.
        * **Chat Functionality:** Enables interaction with the AI to inquire about specific job preferences or seek career advice.
             
        #### Links:
        - [GitHub Repository](https://github.com/samad-ms/JobInsights)
        - [LinkedIn Profile](https://www.linkedin.com/in/abdul-samad-86b158243/)
    """)

    st.write("### Welcome Contributions!")
    st.write("We welcome contributions from the community to improve JobInsights. Whether it's adding new features, fixing bugs, or enhancing documentation, every contribution matters!")

#----------------------------------------------------------------------------------------

def extraction_tab():
    """Data Extraction Tab"""

    st.title("Fetch Job Data.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 Tips", expanded=False):
        st.write(
        """
           * If your job title is uncommon, it's best to skip location details as they further narrow down the dataset.
           * If you're seeking broad information, avoid specifying locations or countries.
           * For faster and better results, Indeed is recommended.
           * Keep in mind that LinkedIn might not always provide job descriptions, which can affect the AI's ability to summarize data accurately.
           * Make sure your search term includes the job role you're looking for.
           * Results wanted are limited to 500 results as of now, the later update will include more quantity.
           * Please make sure you are not clicking the extract button when the process is already running.
        """
        )

    if 'df' not in st.session_state:
        st.session_state.df = None

    col1, col2 = st.columns(2)

    with col1:
        site_name = st.selectbox("Select Site Name: ", ["indeed", "linkedin", "zip_recruiter", "glassdoor"], key="select_site")
    with col2:
        location = st.text_input("Location: Optional", key="location_input", placeholder='Enter Location to search')
    search_term = st.text_input("Search Term", key="search_term_input", placeholder='Eg: Machine Learning Engineer')
    col3, col4 = st.columns(2)
    with col3:
        results_wanted = st.number_input("Results Wanted", key="results_wanted", min_value=1, max_value=500, step=1)
    with col4:
        if site_name == "indeed":
            country = st.text_input("Country: Optional", key="country_input", placeholder='Enter Country')
        else:
            country = st.text_input("Country: Optional", key="country_input", placeholder='Enter Country', disabled=True)

    if site_name == 'linkedin':
        st.warning(
            "Selecting LinkedIn might not provide job descriptions, which could limit the model's ability to generate responses accurately."
            )
    
    # Check if 'desc_string' is not in session_state, initialize it
    if 'desc_string' not in st.session_state:
        st.session_state.desc_string = ""

    st.session_state.search_term = search_term

    extract_button = None
    if search_term  == "":
        extract_button = st.button('Extract Data', key = "extract_button_disabled", disabled=True)
    else:
        extract_button = st.button("Extract Data", key ="extract_button_enabled")

    # Button 1: Extract Data
    if extract_button:
        
        st.session_state.messages = [{'role':'assistant', 'content':'How may I help you?'}]


if __name__ == "__main__":
    st.set_page_config(page_title="JobInsights - AI-driven job seeker")

    feature_tabs = st.sidebar.radio(
        "Features",
        [":rainbow[**Home**]", "**Data Extraction**", "**AI Conversation**"],
        captions=["", "Extract job information as CSV.", "Chat with the AI model to summarize job requirements."]
    )

    if feature_tabs == ":rainbow[**Home**]":
        home_tab()
    elif feature_tabs == "**Data Extraction**":
        extraction_tab()
    elif feature_tabs == "**AI Conversation**":
        chat_tab()

    st.sidebar.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: transparent;
        padding: 10px;
        text-align: left;
        font-size: 15px;
    }
    .footer a {
        text-decoration: none;
        margin-left: 10px;
    }
    </style>
    <div class="footer">
    <a href="https://github.com/samad-ms/JobInsights/issues">Feedback</a>
    <a href="https://github.com/samad-ms/JobInsights">Contributions</a>
    </div>
    """, unsafe_allow_html=True)

