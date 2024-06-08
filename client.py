import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from generate_df import generate_dataframe
from response_generator import model
from preprocess import find_valid_description
from download_csv import get_table_download_link
from response_generator import set_initial_message
from response_generator import chat_with_gemini
from retriver import jd_to_vectorestore,get_response
from ats_utils import *
import uuid

from test import *


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

        with st.status("Extracting Data ... Please Wait", expanded=True):  
            try:
                # Generate the dataframe from details
                st.write("Generating DataFrame ...")
                df = generate_dataframe(site_name, search_term, location, results_wanted, country)
                st.session_state.df = df  # Add to the session_state
                # st.write(df)#---------------------------------------
                # Description String combines all the description generated for the model to summarize   
                total_desc_count = 30
                desc_string = ""
                st.write("Validating Descriptions ...")
                desc_string = find_valid_description(df, total_desc_count, model)
                # st.write(desc_string)#---------------------------------------
                if desc_string:
                    st.session_state.desc_string = desc_string
                else:
                    st.error("No valid description found.")            
                st.success("Data Extraction Complete!")
            except Exception as e:
                st.error(f"Sorry, there is a problem: {e}")
        
        # Passing the initial System Message--------------------------------------
        with st.status("Fetching Descriptions ... Please Wait", expanded = True):
            st.write("Setting Descriptions ... This will take a few moments.")
            try:
                # st.write(st.session_state.desc_string)#--------------------------
                set_initial_message(st.session_state.desc_string) #context set for gemini
                db=jd_to_vectorestore(st.session_state.desc_string)  #context set for gpt-rag
                # st.write(db)#------------------------------------------------------
                if 'db' not in st.session_state:
                    st.session_state.db = db
                st.success("Descriptions Set Successfully!")
            except Exception as e:
                st.error(f"Sorry, there is a problem: {e}")

        with st.spinner("Generating download link ..."):
            try:
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            except:
                pass

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)

if 'messages' not in st.session_state.keys():
    st.session_state.messages = [{'role':'assistant', 'content':'How may I help you?'}]
#----------------------------------------------------------------------------------------

def chat_tab():
    """Chat Interface"""
    st.title("Conversation with AI.")

    with st.expander("💡 Tips"):
        st.write(
        """
        * Ensure to extract the data at least once before engaging in a conversation with the AI to obtain real-time trends and context.
        * Clear prompts lead to better results.
        * Stay on topic to avoid distracting the model from the main subject.

        """
        )

    with st.expander("💡 Example Prompts"):
        try:
            st.write(
                f"""
                    * Identify the top 10 skills mentioned across all job descriptions.
                    * What are the most common experience levels required for jobs?
                    * What are the emerging job trends based on recent job descriptions?
                    * Which locations have the highest demand for this position?
                    * Can you summarize the primary responsibilities mentioned in job descriptions?
                    * Can you identify any patterns related to remote work or flexible schedules in the job descriptions?
                    * Do job descriptions from different regions emphasize different aspects? If so, what are they?
                    * What soft skills are frequently mentioned in job postings?
                    * Provide top resources for mastering various technologies to become proficient in {st.session_state.search_term} within a 3 to 4 month timeframe, along with a comprehensive study plan.
                """
            )
        except:
            pass
        


    if prompt := st.chat_input("Eg: Can you summarize the key insights from the job descriptions?"):
        st.session_state.messages.append({'role':'user', 'content':prompt})
        response = chat_with_gemini(prompt)
        st.session_state.messages.append({'role':'assistant', 'content':response})


    for message in st.session_state.messages:
        if message['role']=='user':
            with st.chat_message('user'):
                    st.write(message['content'])
        if message['role']=='assistant':
            with st.chat_message('assistant'):
                with st.spinner("Thinking ... "):
                    st.write(message['content'])

#----------------------------------------------------------------------------------------                    
def chat_tab_for_gpt():
    """Chat Interface"""
    st.title("Conversation with RAG System.")
    if 'db' in st.session_state:    
        with st.expander("💡 Tips"):
            st.write(
            """
            * Ensure to extract the data at least once before engaging in a conversation with the AI to obtain real-time trends and context.
            * Clear prompts lead to better results.
            * Stay on topic to avoid distracting the model from the main subject.

            """
            )

        with st.expander("💡 Example Prompts"):
            try:
                st.write(
                    f"""
                        * Identify the top 10 skills mentioned across all job descriptions.
                        * What are the most common experience levels required for jobs?
                        * What are the emerging job trends based on recent job descriptions?
                        * Which locations have the highest demand for this position?
                        * Can you summarize the primary responsibilities mentioned in job descriptions?
                        * Can you identify any patterns related to remote work or flexible schedules in the job descriptions?
                        * Do job descriptions from different regions emphasize different aspects? If so, what are they?
                        * What soft skills are frequently mentioned in job postings?
                        * Provide top resources for mastering various technologies to become proficient in {st.session_state.search_term} within a 3 to 4 month timeframe, along with a comprehensive study plan.
                    """
                )
            except:
                pass
        
        if 'rag_history' not in st.session_state:
            st.session_state.rag_history=[{'role':'assistant', 'content':'How may I help you?'}]

        if user_query := st.chat_input("Eg: Can you summarize the key insights from the job descriptions?"):
            # Append user query to chat history
            st.session_state.rag_history.append(HumanMessage(content=user_query))
            
            # Get response from RAG chain and append to chat history
            st.session_state.rag_history.append(AIMessage(content=get_response(db=st.session_state['db'],user_query=user_query,history=st.session_state.rag_history)))
                    
            # Display chat history
        for message in st.session_state.rag_history:
            if isinstance(message, HumanMessage):
                with st.chat_message('Human'):
                    st.write(message.content)
            if isinstance(message, AIMessage):
                with st.chat_message('AI'):
                    with st.spinner("Thinking ... "):
                        st.write(message.content)
    else:
        st.info('Perform data extraction for context setting before engaging with the Retrieval-Augmented Generation (RAG) system.')
#----------------------------------------------------------------------------------------
def ats_tab():
    """Chat Interface"""
    st.title("ATS Score Checker")
    with st.expander("💡 Tips"):
        st.write(
        """
        * Ensure to extract the data to check with real time job discriptions.
        """
        )
    
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    
    if 'job_description' not in st.session_state:
        st.session_state.job_description=''

    jd_submit = st.button('Collect real-time job description')
    if (jd_submit and 'desc_string' in st.session_state) or st.session_state.job_description: #
        job_description= create_job_descriptiion_demo(st.session_state.desc_string,st.session_state.search_term)
    elif jd_submit and 'desc_string' not in st.session_state:
        st.warning('To get real-time job descriptions, extract the data at least once.')
    else:
        job_description = st.text_area("If you have a job description in your hand, please paste the 'JOB DESCRIPTION' here...", key="1")
    
    st.session_state.job_description=job_description
    if st.session_state.job_description != '' and st.session_state.job_description is not None:    
        with st.expander('confirm job discription'):    
            st.write(st.session_state.job_description)
    
    document_count = st.text_input("how many top ranked resume to return back",key="2")
    submit=st.button("Help me with the analysis")


    if submit and st.session_state.job_description:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])
            # st.write(final_docs_list)#-------------------------------------

            # Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

#             #Create embeddings instance
            embeddings=create_embeddings_load_data()
            # st.write(embeddings.embed_query("This is a test document."))#-------------------------------------
        
            #Push data to PINECONE
            ##push_to_pinecone(pinecone_apikey,pinecone_index_name,embeddings,docs):
            vectordb=push_to_pinecone(embeddings=embeddings,docs=final_docs_list)

            # Fecth relavant documents from PINECONE
            relavant_docs=similar_docs(vectordb,st.session_state.job_description,document_count,st.session_state['unique_id'])
            # st.write(relavant_docs)#--------------------------------------------------------------------------

            #Introducing a line separator
            # st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
            
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata["name"])

                #Introducing Expander feature
                with st.expander('Show me Match Score and Content👀'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    # st.write("***"+relavant_docs[item][0].page_content)

                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    # st.write(relavant_docs[item][0])#--------------------
                    # st.write(relavant_docs[item][0].page_content)#--------------------
                    summary = get_summary(relavant_docs[item][0])['output_text']
                    st.write("**Summary** : "+str(summary))

            st.success("Hope I was able to save your time⏰")
        
#----------------------------------------------------------------------------------------


if __name__ == "__main__":
    st.set_page_config(page_title="JobInsights - AI-driven job seeker",page_icon='💼')

    feature_tabs = st.sidebar.radio(
        "Features",
        [":rainbow[**Home**]", "**Data Extraction**", "**AI Conversation**","**RAG System**","**ATS System**"],
        captions=["", "Extract job information as CSV.", "Chat with the AI model to summarize job requirements.","same as AI Conversation but with vector-database","AI tool for efficient and accurate resume screening in ATS"]
    )

    if feature_tabs == ":rainbow[**Home**]":
        home_tab()
    elif feature_tabs == "**Data Extraction**":
        extraction_tab()
    elif feature_tabs == "**AI Conversation**":
        chat_tab()
    elif feature_tabs == "**RAG System**":
        chat_tab_for_gpt()
    elif feature_tabs == "**ATS System**":
        ats_tab()

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

