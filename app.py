import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image 
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

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:       
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vecotr_store(text_chunks):
#     emebeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectore_store = FAISS.from_texts(text_chunks, embedding=emebeddings)
#     vectore_store.save_local("Training_Data")
    


def get_conversational_chain():
    # prompt_template = """
    # You are the administrator of a physician group seeking GUIDE reimbursement. Your practice, specializing in dementia care, geriatrics, and gerontology, includes dementia care specialists, geriatricians, gerontologists, and a care navigator. To strategically leverage technology and maximize your chances of success in meeting GUIDE program requirements, you're helping the organisation to fill up the form. You have been asked to provide a tailored response to the following question, ensuring alignment with GUIDE eligibility criteria, program priorities, and the specific characteristics of your organization. Keep your answer concise and impactful, within 3000 characters. Take help of the context and organisation type to answer the question.\n\n
    # Context: {context}\n
    # Question: {question}\n
    # organisation_type: {organisation_type}\n
    # """
    prompt_template = """
    You are the administrator of a physician group seeking GUIDE reimbursement. Our practice specializes in dementia care, geriatrics, and gerontology, bringing together dementia care specialists, geriatricians, gerontologists, and a care navigator. To maximize your chances of success, we're collaboratively filling out the form, leveraging technology strategically to meet GUIDE program requirements. 
    Given the following context, please provide a tailored response to this question, ensuring alignment with GUIDE's eligibility criteria, program priorities, and our organization's specific characteristics. Keep your answer concise and impactful, within 3000 characters.
    Organization Type: {organisation_type}\n
    Context: {context} \n
    Question: {question} \n
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question", "organisation_type"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain




def user_input(user_question,organisation_type):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("Training_Data", embeddings)
    docs = new_db.similarity_search(user_question, k=4)

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    # chain = load_qa_chain(llm=llm, chain_type="stuff")
    chain = get_conversational_chain()

    
    # response = chain(        input_documents=docs, question= user_question)
    response = chain(
        {"input_documents":docs, "question": user_question, "organisation_type": organisation_type }
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])




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
                <img src="https://assets-global.website-files.com/61615716b8ea6250c3ff2ece/638cf97ede359004216a3c4e_Evva%20logo_1.png" width=100>
            </a>
            """, unsafe_allow_html=True)

        with col1:
            st.header("Get your answers about the CMS GUIDE Program")

    organisation_type = st.text_input("Type of Organization")
    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question, organisation_type)

    



if __name__ == "__main__":
    main()