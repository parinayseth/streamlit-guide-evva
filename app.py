import streamlit as st
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

import os
from langchain_community.chat_models import ChatOpenAI

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain.vectorstores.faiss import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_conversational_chain():
    prompt_template = """
    You are an expert in the CMS GUIDE program for dementia care. Please respond to the user's question from your available knowledge in easy to understand language. Consider user's organization type in framing your response, if applicable.\n\n
    Organization Type: {organisation_type}\n
    Context: {context} \n
    Question: {question} \n
    Make sure the answer is properly formatted and grammatically correct. Dont give responses like - "Based on provided information" in your answers \n
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question", "organisation_type"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain




def user_input(user_question,organisation_type):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    new_db = FAISS.load_local("Training_Data", embeddings)
    docs = new_db.similarity_search(user_question, k=3)

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    # chain = load_qa_chain(llm=llm, chain_type="stuff")
    chain = get_conversational_chain()

    
    # response = chain(        input_documents=docs, question= user_question)
    response = chain(
        {"input_documents":docs, "question": user_question, "organisation_type": organisation_type }
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])
    return response["output_text"]

def save_excel(organisation_name, email, organisation_type, user_question, response):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%d/%m/%Y")
    credentials_path = "evva-health-project-fb14724414f1.json"
    spreadsheet_name = 'Evva Health Guide Program'
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file(credentials_path, scopes=scope)
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open(spreadsheet_name)
    worksheet = spreadsheet.get_worksheet(0)
    row_data = [current_date, current_time, organisation_name, email, organisation_type, user_question, response]
    worksheet.append_row(row_data)
    

    


    
def main():
    st.set_page_config("Welcome to CMS GUIDE Program (Evva Health)")

    st.markdown("""
<style>
.columns-container {
    display: flex;
}
.col1 {
    flex: 5.5;  /* Adjust this value for column width ratio */
}
.col2 {
    flex: 1;  /* Adjust this value for column width ratio */
}
.logo-img {
    position: relative; /* Adjust positioning as needed */
}
</style>
""", unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([5.5, 1])  # Create columns with a 5.5:1 width ratio

        with col2:
            link = "https://www.evva360.com/"  # Replace with your desired link
            st.markdown(f"""
            <a href="{link}" class="logo-img">
                <img src="https://i.postimg.cc/1tdqbnQf/logo.jpg" width=105>
            </a>
            """, unsafe_allow_html=True)

        with col1:
            st.header("Get your answers about the CMS GUIDE Program")
    organisation_name = st.text_input("What is the name of your organization?")
    email = st.text_input("What is your email address?")
    organisation_type = st.text_input("What is the type of your organization? (e.g., physician group, individual practice, health system, etc.)")
    user_question = st.text_input("Ask a Question")

    
    if st.button("Submit"):
        if user_question == "":
            st.write("Please enter a question")
        if organisation_type == "":
            st.write("Please enter a type of organization")
        with st.spinner("Processing..."):
            result =  user_input(user_question, organisation_type)
            save_excel(organisation_name, email, organisation_type, user_question, result)

    



if __name__ == "__main__":
    main()