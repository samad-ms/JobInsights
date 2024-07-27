import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from generate_df import generate_dataframe
from response_generator import model
from preprocess import *
from download_csv import get_table_download_link
from response_generator import set_initial_message
from response_generator import chat_with_gemini
from retriver import jd_to_vectorestore,get_response
from ats_utils import *
from email_utils import *
from visualization_utils import *
import uuid
from dotenv import load_dotenv
load_dotenv()

#----------------------------------------------------------------------------------------
def home_tab():
    """ Home Tab """
    st.title("🔍 Welcome to JobInsights!")
    st.info(" 🔝⬅️ If you wish to navigate to the features, you can directly click on the arrow located in the top-left corner, then select the particular feature you wish to use.")
    st.write("""
        #### What is it?

        **JobInsights** is an AI-powered application designed to streamline the job hunting process by aggregating real-time data from popular job platforms such as Indeed, LinkedIn, Glassdoor, and ZipRecruiter. The application provides personalized and comprehensive job summaries, insights, ways to optimize resumes, email generation during job searches, and many other features for each query.

        #### How it Works?

        **JobInsights** utilizes advanced technologies including GPT-3.5-turbo, Gemini, LIDA (Microsoft), LangChain, and vector databases to analyze and comprehend hundreds of job descriptions. By leveraging this understanding, the platform offers various features and strategies to secure job interviews and succeed in them. The automated process saves significant time by eliminating the need for rigorous manual work.

        #### Features

        * **Data Extraction:** Extract real-time job information from popular websites.
        * **Understand the Job Market:** Gain insights into the real-time job market through trend analysis and visualizations.
        * **AI Conversation:** A general assistant that summarizes job requirements and answers general queries.
        * **RAG AI System:** A specialized assistant that responds by considering the real-time job market as context.
        * **ATS Score Checker and Resume Optimization:** Generates concise job descriptions and conducts similarity searches with resumes to identify the top matching resume between job seekers and available positions. It provides keywords to optimize the resume and improve the match.
        * **Email Generator:** Create customized emails by extracting relevant details of candidates from resumes or portfolio links, enabling users to quickly and efficiently reach out to potential employers.
        * **Additional Features:** JobInsights is continuously evolving to offer more exciting features. Stay tuned for:
                - Building resumes from raw text or portfolios that match real-time job requirements.
                - AI-driven interview preparation using resumes and real-time job requirements for specific job roles.
                - AI for preparing learning materials and interview preparation.
                - And many more exciting features to come!

        #### Technologies

        JobInsights is built using Python and incorporates specialized scraping libraries for data extraction. The user interface is developed using Streamlit for an interactive experience. The application leverages the power of GPT-3.5-turbo for natural language processing, Gemini 1.0 Pro for deep market analysis, LIDA by Microsoft for Visualization tasks, LangChain for efficient data processing, and Pinecone,FAISS as the vector database for storing and retrieving data efficiently.

        #### Contributing

        Contributions are welcome! If you have any suggestions or improvements, please feel free to open a pull request or an issue.

        #### License

        This project is licensed under the MIT License.

        #### Links
        - **Application:** [JobInsights](https://job-insights.streamlit.app/)
        - **Report:** [Project Report](https://abduls-organization-13.gitbook.io/abduls-portfolio/projects/job-insights)
        - **GitHub Repository:** [JobInsights GitHub](https://github.com/samad-ms/JobInsights)
        - **LinkedIn Profile:** [Abdul Samad on LinkedIn](https://www.linkedin.com/in/abdul-samad-86b158243/)
    """)

    st.write("### Welcome Contributions!")
    st.write("We welcome contributions from the community to improve JobInsights. Whether it's adding new features, fixing bugs, or enhancing documentation, every contribution matters!")
#----------------------------------------------------------------------------------------
def extraction_tab():
    """Data Extraction Tab"""

    st.title("Fetch Job Data.")
    st.markdown("<br>", unsafe_allow_html=True)


    with st.expander("🗺️ Step-by-Step Guide", expanded=False):
        st.write(
        """
            Welcome to the Data Extraction feature! Follow these steps to efficiently extract job data from the platform.

            1. **Select Site Name:**
            - Options:
                - Indeed
                - LinkedIn
                - Glassdoor
                - Other 
            - Instruction: Choose the job site from the list.

            2. **Location (Optional):**
            - Prompt: Enter the location where you want to search for jobs.
            - Example: "Bangalore, India"
            - Instruction: If you wish to specify a location, enter the city, state, or region. This step is optional and can be left blank to search nationwide.

            3. **Search Term:**
            - Prompt: Enter the job title or keywords related to the job you are searching for.
            - Example: "Machine Learning Engineer (Entry Level)"
            - Instruction: Input the specific job title or keywords that match the jobs you are interested in.

            4. **Results Wanted:**
            - Prompt: Specify the number of job listings you want to retrieve.
            - Example: "1"
            - Instruction: Enter the desired number of results. This determines how many job listings will be extracted.

            5. **Country (Optional):**
            - Prompt: Enter the country for the job search.
            - Example: "India"
            - Instruction: If you want to narrow down your search to a specific country, enter the country name. This step is optional.

            **Example Workflow:**
            - Select Site Name: LinkedIn
            - Location (Optional): "Bangalore, India"
            - Search Term: "Machine Learning Engineer (Entry Level)"
            - Results Wanted: "1"
            - Country (Optional): "India"        
            
            Click 'Download CSV File' to download the data and start applying.
            """
        )
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
    search_term = st.text_input("Search Term", key="search_term_input", placeholder='Eg: Machine Learning Engineer (Entry Level)')
    col3, col4 = st.columns(2)
    with col3:
        results_wanted = st.number_input("Results Wanted", key="results_wanted", min_value=10, max_value=500, step=1)
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
                # desc_string_rag = find_valid_description(df, total_desc_count, model)
                # st.write(desc_string)#---------------------------------------
                desc_string_rag=''
                if 'desc_string_rag' not in st.session_state:
                    st.session_state.desc_string_rag = ""
                desc_string_rag,relevant_indices=remove_unnecessary_info_from_job_description(st.session_state.search_term,st.session_state.df)
                # desc_string_rag=format_job_descriptions(st.session_state.df)
                # st.write(desc_string_rag)#---------------------------------------
                if desc_string_rag:
                    st.session_state.desc_string_rag = desc_string_rag
                    if len(relevant_indices)<=2:
                        st.info("Insufficient job descriptions may lead to incomplete analyses and limit the depth of insights gained.")
                else:
                    st.error("No valid description found.")            

                # if desc_string:
                #     st.session_state.desc_string = desc_string
                # else:
                #     st.error("No valid description found.")            
                st.success("Data Extraction Complete!")
            except Exception as e:
                st.error(f"Sorry, there is a problem: {e}")
        
        # Passing the initial System Message--------------------------------------
        with st.status("Fetching Descriptions ... Please Wait", expanded = True):
            st.write("Setting Descriptions ... This will take a few moments.")
            try:
                # st.write(st.session_state.desc_string)#--------------------------
                # set_initial_message(st.session_state.desc_string) #context set for gemini
                db=jd_to_vectorestore(st.session_state.desc_string_rag)  #context set for gpt-rag

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
        st.warning('Perform data extraction for context setting before engaging with the Retrieval-Augmented Generation (RAG) system.')
#----------------------------------------------------------------------------------------
def ats_tab():
    """Chat Interface"""
    st.title("ATS Score Checker")
    with st.expander("💡 **Step-by-Step Guide for Using the ATS Score Checker**"):
        st.write(
        """

        **Professional Tip:** Enhance your job application strategy by creating multiple versions of your resume with diverse variations. This approach allows you to tailor your resume to specific job requirements, increasing your chances of standing out to potential employers. 

        1. **Upload Resumes:**
        - Ensure all files are in PDF format.
        - Drag and drop your resume files into the designated area.

        2. **Job Description:**
        - If you possess a job description, insert it into the 'JOB DESCRIPTION' field.
        - In the absence of a job description, click 'Collect Real-Time Job Description' to retrieve one.
        - Confirm the job description by expanding the expander.

        3. **Submit:**
        - Initiate the analysis process by clicking 'Help me with the analysis'.

        **Score, Summary, and Keywords:**

        Upon submission, receive feedback on your best resume version, including a score, summary, and keywords. Incorporate these keywords into your resume to optimize its effectiveness and boost your score.
        
    """
        )
    
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    
    if 'job_description' not in st.session_state:
        st.session_state.job_description=''
    job_description = ''
    
    with st.expander("💡 **Are You Unsure About the Button?**"):
        st.write(
        """
            The "Collect Real-Time Job Description" button quickly retrieves a job description by combining various sources, making it useful for users without a specific description on hand. This generated description contains a wide range of keywords, enhancing the resume's compatibility with different job descriptions and improving its acceptance rate.
        """)

    jd_submit = st.button('Click here to collect a real-time job description if you do not have one available.')
    
    try:
        if jd_submit:
            if 'db' in st.session_state:
                job_description= create_job_descriptiion_demo(st.session_state.desc_string,st.session_state.search_term if "search_term" in st.session_state else '')
            else:
                st.warning('To get real-time job descriptions, extract the data at least once.')
        else:
            job_description = st.text_area("Kindly paste the job description here if you have one available..", key="1")
    except Exception as e:
        st.error(f"To get real-time job descriptions, extract the data at least once.")


    if job_description != '' and job_description is not None:
        st.session_state.job_description=job_description
    if st.session_state.job_description != '' and st.session_state.job_description is not None:    
        with st.expander('**confirm job discription**'):    
            st.write(st.session_state.job_description)
    
    # document_count = st.text_input("how many top ranked resume to return back",key="2")
    document_count= 1
    submit=st.button("Help me with the analysis")


    if submit:
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

            #Introducing a line separator
            # st.write(":heavy_minus_sign:" * 30)
            if len(relavant_docs)>0:
                for item in range(len(relavant_docs)):
                    st.subheader("👉 "+str(item+1))

                    #Displaying Filepath
                    st.write("**File** : "+relavant_docs[item][0].metadata["name"])

                    #Introducing Expander feature
                    with st.expander('Show me Match Score and Content👀'): 
                        st.info(f"Match Score: {str(int(100*(relavant_docs[item][1])))} %")
                        # st.write("***"+relavant_docs[item][0].page_content)

                        #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                        # st.write(relavant_docs[item][0])#--------------------
                        # st.write(relavant_docs[item][0].page_content)#--------------------
                        summary = get_summary(relavant_docs[item][0])['output_text']
                        keywords = get_keywords_to_optimize_resume(str(relavant_docs[item][0].page_content),st.session_state.job_description)
                        st.write("**Summary** : "+str(summary))
                        st.write("**Keywords to Include in Your Resume to Optimize and Match Job Description** : "+str(keywords))

                st.success("Hope I help you to find the best resume⏰")
            else:
                st.write('please click "Help me with the analysis"')
#----------------------------------------------------------------------------------------
def email_tab():
    emails = """
        **1. Application Email**:
        - **Purpose**: To apply for a specific job opening.
        - **Content**: Includes a cover letter, resume, and any additional required documents.
        - **Example Subject**: Application for [Job Title] Position

        **2. Follow-Up Email**:
        - **Purpose**: To inquire about the status of an application or express continued interest.
        - **Content**: Polite inquiry about application status, reiteration of interest, and availability for further discussions.
        - **Example Subject**: Follow-Up on Job Application for [Job Title] Position

        **3. Thank-You Email**:
        - **Purpose**: To thank the interviewer(s) for the opportunity to interview.
        - **Content**: Expresses gratitude, reiterates interest, and may include a brief mention of something discussed during the interview.
        - **Example Subject**: Thank You for the Interview - [Job Title] Position

        **4. Acceptance Email**:
        - **Purpose**: To formally accept a job offer.
        - **Content**: Expresses gratitude for the offer, confirms acceptance, and may include any requested information, such as start date availability.
        - **Example Subject**: Acceptance of Job Offer for [Job Title] Position

        **5. Withdrawal Email**:
        - **Purpose**: To withdraw an application from consideration.
        - **Content**: States the decision to withdraw, expresses appreciation for the opportunity, and may include a brief reason for withdrawal (optional).
        - **Example Subject**: Withdrawal of Application for [Job Title] Position

        **6. Update Email**:
        - **Purpose**: To provide updated information or qualifications.
        - **Content**: Clearly states the update (e.g., completed course, obtained certification) and expresses continued interest in the position.
        - **Example Subject**: Updated Information for [Job Title] Application
        """


    st.header("Generate Emails 📧")

    with st.expander(f"💡 **Types of Emails Used in the Job Hunting Process**"):
        st.write(emails)
    with st.expander(f"💡 **Step-by-Step Guide for Using the Email Generation**"):
        st.write("""

            1. **Select Email Type:**
            - Choose the type of email you want to generate from the options provided.

            2. **Enter the Email Details:**
            - Input the email topic (e.g., Application for Machine Learning Engineer Role at ExampleTech).

            3. **Enter Your Name:**
            - Enter your name.

            4. **Recipient Information:**
            - Enter the recipient's name.

            5. **Gather Additional Professional Information:**
            - Choose one of the following options:
                - Upload your resume for gathering additional professional information.
                - Provide your portfolio link for gathering additional professional information.
                - Write your relevant information.
                - No mention.

            6. **Enter Signature:**
            - Input your contact details (job title, company, phone number, etc.).

            7. **Generate:**
            - Click the Generate button to generate the email.

            8. **Adjust and Regenerate:**
            - If you are not satisfied with the result, click the Generate button again or tweak the Additional Professional and Email Details a little and try again.

            """)

    # email_type = st.selectbox("Select Email Type: ", ["Application Email", "Follow-Up Email", "Thank-You Email", "Acceptance Email", "Withdrawal Email", "Update Email", "Cover Letter", "Other"], key="select_email_type")

    form_input = st.text_area('Enter the email topic',placeholder='Application for Machine Learning Engineer Role at ExampleTech',height=200)
    # Creating columns for the UI - To receive inputs from user
    col1, col2 = st.columns([10, 10])
    with col1:
        email_sender = st.text_input('Sender Name')
    with col2:
        email_recipient = st.text_input('Recipient Name')
    

    candidate_details_text=""
    
    candidate_details_fetch_from = st.selectbox("Gather additional professional information about you from:", ["Resume", "Portfolio website","Write","Not mentioned"], key="candidate_details")
    if candidate_details_fetch_from=='Resume':
        resume_pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
        # st.session_state['unique_id']=uuid.uuid4().hex
        # candidate_details_text=create_docs(resume_pdf,st.session_state['unique_id']) #if multiple files comes
        if candidate_details_text:=create_docs(resume_pdf,1):
            candidate_details_text=candidate_details_text[0].page_content
    elif candidate_details_fetch_from=='Portfolio website':
        if website_url := st.text_input('provide your website url , which have relevent information about you..'):
            candidate_details_text=get_candidate_details(website_url=website_url,candidate_name=email_sender)
    elif candidate_details_fetch_from=="Write":
        candidate_details_text=st.text_area('write about yourself',placeholder="Ex: self-taught Machine Learning Engineer at Brototype, Kochi, specializing in developing AI-powered solutions for job search optimization and demonstrating proficiency in Python, Machine Learning, Deep Learning, NLP, and computer vision.")               
    else:
        candidate_details_text=""

    if candidate_details_text:
        if ("candidate_details_text" not in st.session_state) or candidate_details_text:
            st.session_state.candidate_details_text=candidate_details_text
        with st.expander("**confirm candidate details**"):
            st.write(st.session_state.candidate_details_text)


    company_details_text=''
    company_details_fetch_from = st.selectbox("Gather additional information about the company from:", ["website","Write","Not mentioned"], key="company_details")
    if company_details_fetch_from=='website':
        if website_url := st.text_input("provide the URL for the 'About Us' page on your company's website."):
            company_details_text=get_company_details(website_url=website_url)
    elif company_details_fetch_from=="Write":
        company_details_text = st.text_area('Company Details', placeholder="Ex: Established in 2010, AlphaTech is a renowned leader in AI and machine learning solutions, headquartered in City X. Our team of 200+ experts specializes in developing cutting-edge applications for data analytics, natural language processing (NLP), and computer vision. AlphaTech's innovative products have been featured in industry publications and recognized for their impact on optimizing business processes and customer engagement.")
    else:
        company_details_text=""

    if company_details_text:
        if ("company_details_text" not in st.session_state) or company_details_text:
            st.session_state.company_details_text = company_details_text
        with st.expander("**Confirm Company Details**"):
            st.write(st.session_state.company_details_text)




    Signature = st.text_area('Signature: your contact details (job title, company, phone number, etc.)',
                            placeholder="""Abdul Samad \nMachine Learning Engineer \nExampleTech \nsamad.example@gmail.com \n+91-1234567890 \n """
                            ,height=100)
    # st.write(Signature)
    submit = st.button("Generate")

    # When 'Generate' button is clicked, execute the below code
    response=None
    if submit:
        with st.spinner("Thinking ... "):
            response=get_email_Response(form_input, email_sender, email_recipient,Signature,candidate_details_text=candidate_details_text,company_details_text=company_details_text)
        # clipboard.copy(f"{response.content}")
        # st.success("Text copied to clipboard!")
            st.write(response.content)
#----------------------------------------------------------------------------------------
def realtime_job_market_visualization_tab():

    st.subheader("Understand the real-time job market 📈")
    with st.expander(f"💡 **Tips**"):
        st.write("Extract more data to get better results.")

    if 'db' in st.session_state:
        button=st.button("Generate the insights")
        if button:
            path_to_save = "filename.csv"
            st.session_state.df.to_csv(path_to_save, index=False)
            # with open(path_to_save, "wb") as f:
            #     f.write(st.session_state.df.getvalue())
            

            # st.write(auto_summarizer())
            try:
                goals_and_imgs=auto_summarizer()
                for item in goals_and_imgs:
                    st.write(item[0])
                    st.write(item[1])
                    st.write(item[2])
                    st.image(item[3])
                # st.write(chat_with_gemini('What are the key insights in the job market? \n What are the top skills to learn for this particular job role? How can job seekers improve their chances of success in this field?'))
            except:
                st.warning('The available dataset is insufficient; it lacks the necessary information to generate insights and visualizations. Please extract more data.')
    else:
        st.warning("Please ensure that you have extracted the data at least once before interacting with this feature")

#----------------------------------------------------------------------------------------
def Additional_features():
    """ Additional features Tab """
    st.title("🔍 Additional Features, Coming Soon...!")
    st.write("""
        1) Building resumes from raw text or portfolios that match real-time job requirements.
        2) AI-driven interview preparation using resumes and real-time job requirements for specific job roles.
        3) AI for preparing learning materials and interview preparation.
        4) And many more exciting features to come!

        Thank you for visiting! Share your thoughts and ideas to integrate, and also, welcome to contributions and debugging!
    """)
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="JobInsights - AI-driven job seeker",page_icon='💼')

    feature_tabs = st.sidebar.radio(
        "Features",
        [
            ":rainbow[**Home**]",
            "**Job Data Extraction**",
            # "**General AI Assistant**",
            "**Specilized AI Assistant**",
            "**ATS System and Resume Optimization**",
            "**Email Generator**",
            "**Job Market Analysis**",
            "**Additional Features**"
        ],
        captions=[
            "",
            "Extract job information from websites.",
            # "General AI for job requirements and queries.",
            "Specialized AI with real-time job market as context.",
            "Generates job descriptions and optimizes resumes.",
            "Create customized emails.",
            "Gain insights through trend analysis.",
            ""
        ]

    )

    if feature_tabs == ":rainbow[**Home**]":
        home_tab()
    elif feature_tabs == "**Job Data Extraction**":
        extraction_tab()
    # elif feature_tabs == "**General AI Assistant**":
    #     chat_tab()
    elif feature_tabs == "**Specilized AI Assistant**":
        chat_tab_for_gpt()
    elif feature_tabs == "**ATS System and Resume Optimization**":
        ats_tab()
    elif feature_tabs == "**Email Generator**":
        email_tab()
    elif feature_tabs == "**Job Market Analysis**":
        realtime_job_market_visualization_tab()
    elif feature_tabs == "**Additional Features**":
        Additional_features()

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
#----------------------------------------------------------------------------------------
