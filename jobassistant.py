# import langchain related libraries

import os
import streamlit as st
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import pathlib
# Load environment variables from .env file
load_dotenv()

# # Access the OpenAI API key from environment
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY not found in .env file"

# streamlit initializations
st.set_page_config(page_title="Job Assistant", layout="centered")
st.title("🤖 Job Assistant")

# Read the resume text from file
RESUME_FILE = pathlib.Path(__file__).parent / "resume.txt"
resume_text = RESUME_FILE.read_text()

def display_job_output(data):
    st.title("📄 AI Resume Analysis")
    
    # --- Top Row: Match Score and Key Skills ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Resume Match Score", value=f"{data.resume_matching}%")
        
        # Color-coded progress bar
        if data.resume_matching >= 80:
            st.progress(data.resume_matching / 100, "High Match")
        elif data.resume_matching >= 50:
            st.progress(data.resume_matching / 100, "Moderate Match")
        else:
            st.progress(data.resume_matching / 100, "Low Match")

    with col2:
        st.subheader("Key Skills Identified")
        # Display skills as tags using a loop
        skills_html = "".join(
            [f'<span style="background-color: #0e1117; color: #ff4b4b; padding: 5px 10px; border-radius: 15px; margin: 5px; border: 1px solid #ff4b4b; display: inline-block;">{skill}</span>' 
             for skill in data.key_skills]
        )
        st.markdown(skills_html, unsafe_allow_html=True)

    st.divider()

    # --- Middle Section: Messages ---
    st.subheader("📧 Outreach Templates")
    tab1, tab2 = st.tabs(["Recruiter Message", "Networking Message"])
    
    with tab1:
        st.info("Personalized message for the hiring team.")
        st.text_area("Recruiter Message", data.recruiter_message, height=200)
        st.button("Copy Recruiter Message", on_click=lambda: st.write("Copied! (Mock)"))

    with tab2:
        st.info("LinkedIn/Internal referral message.")
        st.text_area("Employee Outreach", data.current_employee_message, height=150)
    
    st.divider()

    # --- Bottom Section: Resume Preview ---
    with st.expander("🔍 View Generated/Sample Resume"):
        st.code(data.sample_resume, language="markdown")


class JobDescriptionOutput(BaseModel):
    key_skills: list[str] = Field(description="The key skills required for the job")
    resume_matching: int = Field(description="The percentage of matching with the resume text (0-100)")
    sample_resume: str = Field(description="The Sample resume of perfect candidate for this role.")
    recruiter_message: str = Field(description="A professional message for the recruiter showing eagerness to join the company with matching experience and skills. Include the company name from the job description.")
    current_employee_message: str = Field(description="A friendly message for a current employee (Software Engineer) at the company. Mention that you are an Illinois Tech alumni and ask for any help or advice. Message should be short and professional so that busy person can read and reply")


agent = create_agent(
    model="openai:gpt-4o-mini",  # Use a valid OpenAI model name
    response_format=JobDescriptionOutput,  # Auto-selects ProviderStrategy
)

user_input = st.text_area("Enter your request:", placeholder="Job Description")

combined_input = f"""
            Please analyze the following:
            
            JOB DESCRIPTION:
            {user_input}
            
            RESUME TEXT:
            {resume_text}
        
            """

if st.button("Run Agent"):
    if user_input:
        
        with st.spinner("Thinking..."):

            promptMessages = [  # Wrap in dict with "messages" key
                SystemMessage(content="You are a job application assistant..."),
                HumanMessage(content="Analyze the following job description and the resume text"),
                HumanMessage(content=combined_input)
            ]
            result = agent.invoke({"messages": promptMessages})
            output = result["structured_response"]
            display_job_output(output)
            
            
    else:
        st.warning("Please enter a prompt first.")


import streamlit as st


# To run this, you'd pass your Pydantic object to the function:
# display_job_output(your_pydantic_object)
