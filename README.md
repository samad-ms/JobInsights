# JobInsights
![Python](https://img.shields.io/badge/python-v3.11.3-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-v3.5-blue)
![LangChain](https://img.shields.io/badge/LangChain-Icon-green)
![Gemini 1.0 Pro](https://img.shields.io/badge/Gemini%201.0%20Pro-blue)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0.0-green)
![Pinecone](https://img.shields.io/badge/Pinecone-Icon-green)
![RAG](https://img.shields.io/badge/RAG-Icon-green)


## Application | Report | Source Code

- **Application:** https://job-insights.streamlit.app/
- **Report:** https://abduls-organization-13.gitbook.io/abduls-portfolio/projects/job-insights

## Description
Developed JobInsights, an AI-powered application designed to streamline the job hunting process by aggregating real-time data from popular job platforms such as Indeed, LinkedIn, Glassdoor, and ZipRecruiter. The application utilizes advanced technologies including GPT-3.5-turbo, Gemini 1.0 Pro, and LangChain for comprehensive job market analysis and optimization.

## Features
- **Data Extraction**: Extract real-time job information and visualize job market trends for easy analysis and comparison and context setting for AI models.
- **AI Conversation (RAG)**: Utilizing Real-time Assistive Guidance (RAG), JobInsights summarizes job requirements and provides personalized responses based on the latest market data, helping users tailor their applications effectively.
- **ATS Score Checker (RAG)**: JobInsights generates concise job descriptions and conducts similarity searches with resumes to optimize the match between job seekers and available positions, enhancing the chances of successful application.
- **Email Generator**: Create customized emails by extracting relevant details from resumes or portfolio links, enabling users to quickly and efficiently reach out to potential employers.

## Technologies
JobInsights is built using Python and incorporates specialized scraping libraries for data extraction. The user interface is developed using Streamlit for an interactive experience. The application leverages the power of GPT-3.5-turbo for natural language processing, Gemini 1.0 Pro for deep market analysis, and LangChain for efficient data processing. Additionally, Agents are used for task automation, while Pinecone serves as the vector database for storing and retrieving data efficiently.

- ## Usage

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run src/client.py
   ```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to open a pull request or an issue.

## License

This project is licensed under the MIT License.
