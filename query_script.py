import re
import time
import json
import spacy
import pandas as pd
import pickle
import faiss
import streamlit as st
from sqlalchemy import create_engine, inspect
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv
import os
load_dotenv()


nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

engine = create_engine(os.environ['DB'])
inspector = inspect(engine)

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result, execution_time
    return wrapper

def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if token.is_alpha]
    text = ' '.join(lemmatized_words)
    text = re.sub(r'[^,\w\s]', '', text)
    text = re.sub(r'_', ' ', text)
    return text

def cache_metadata(schema_df):
    schema_df.to_json('metadata.json', orient='records')
    print("Cached schema metadata.")

def load_cached_metadata():
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
            print("Loaded cached metadata.")
            return pd.DataFrame(metadata)
    except FileNotFoundError:
        print("No cached metadata found.")
        return None

def cache_tfidf_vectors(vectorizer, tfidf_matrix, table_descriptions):
    with open('tfidf_cache.pkl', 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix, table_descriptions), f)
    print("TF-IDF vectors cached.")

def load_cached_tfidf_vectors():
    try:
        with open('tfidf_cache.pkl', 'rb') as f:
            vectorizer, tfidf_matrix, table_descriptions = pickle.load(f)
        print("Loaded cached TF-IDF vectors.")
        return vectorizer, tfidf_matrix, table_descriptions
    except FileNotFoundError:
        print("No cached TF-IDF vectors found.")
        return None, None, None

exclude_tables = []

def get_schema_metadata():
    metadata = []
    table_names = inspector.get_table_names()
    table_names = [table for table in table_names if table not in exclude_tables]

    for table in table_names:
        columns = inspector.get_columns(table)
        column_names = [col['name'] for col in columns]
        description = f"Table: {table}, Columns: {', '.join(column_names)}"
        preprocessed_description = preprocess_text(description)
        metadata.append({'table_name': table, 'description': preprocessed_description})

    schema_df = pd.DataFrame(metadata)
    print(f"Schema metadata generated: {schema_df}")

    cache_metadata(schema_df)
    
    table_descriptions = schema_df['description'].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(table_descriptions)
    
    cache_tfidf_vectors(vectorizer, tfidf_matrix, table_descriptions)

    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tfidf_matrix.toarray().astype('float32'))

    return schema_df, vectorizer, index

def score_table_relevance(query, index, vectorizer, k=8):
    preprocessed_query = preprocess_text(query)
    query_tfidf = vectorizer.transform([preprocessed_query]).toarray().astype('float32')

    distances, indices = index.search(query_tfidf, k)
    return indices[0], distances[0]

def get_relevant_tables_from_cache(query, threshold=1.5):
    schema_df = load_cached_metadata()
    
    if schema_df is None:
        schema_df, vectorizer, index = get_schema_metadata()
    else:
        vectorizer, tfidf_matrix, table_descriptions = load_cached_tfidf_vectors()
        if vectorizer is None or tfidf_matrix is None:
            schema_df, vectorizer, index = get_schema_metadata()
        else:
            dimension = tfidf_matrix.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(tfidf_matrix.toarray().astype('float32'))

    relevant_indices, distances = score_table_relevance(query, index, vectorizer)
    
    relevant_tables = [
        schema_df['table_name'][i] for i, distance in zip(relevant_indices, distances) 
        if distance < threshold 
    ]
    
    print(f"Relevant tables for query '{query}' based on threshold {threshold}: {relevant_tables}")
    return relevant_tables

def create_filtered_database(relevant_tables):
    relevant_tables = [table for table in relevant_tables if table not in exclude_tables]
    filtered_db = SQLDatabase(engine, include_tables=relevant_tables)
    print(f"Filtered database created with tables: {relevant_tables}")
    return filtered_db

@timer_decorator
def execute_query_on_relevant_tables(query, llm):
    relevant_tables = get_relevant_tables_from_cache(query)
    filtered_db = create_filtered_database(relevant_tables)
    toolkit = SQLDatabaseToolkit(db=filtered_db, llm=llm)
    agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    
    result = agent_executor.run(query)
    print(f"Query Result: {result}")
    return result 