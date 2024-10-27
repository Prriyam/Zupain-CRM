# app.py

import streamlit as st
from langchain.chat_models import ChatOpenAI
from query_script import execute_query_on_relevant_tables
from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(openai_api_key = openai_api_key )

st.title("Zupain CRM")
st.write("Ask a question to retrieve information from your database.")

query = st.text_input("Enter your question:")

if st.button("Submit Query"):
    if query:
        with st.spinner("Querying the database..."):
            result,execution_time = execute_query_on_relevant_tables(query, llm)
            st.write(f"Query Result: {result}")
            st.write(f"Execution Time: {execution_time:.2f} seconds") 
    else:
        st.write("Please enter a question.")
