import streamlit as st
import json
import time
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
import torch
from tqdm.auto import tqdm
from pinecone_text.sparse import BM25Encoder

page_bg_img = """
<style>
[data-testid="stSidebar"] > div:first-child {
    background-color: #FFC300;
    color: #000000;

}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
<style>
    div[data-baseweb="input"] > div {
        background-color: #FCFCFC;
        color: black;
        border: 2px solid black;
        border-radius: 5px;
        padding: 20px;
        font-size: 25px;
    }
    div[data-baseweb="input"] > div:focus {
        outline: none;
        border-color: black;
    }
    div[data-baseweb="slider"] > div > div:first-child {
        background-color: white;
    }
    div[data-baseweb="slider"] > div > div:first-child > div {
        background-color: black;
    }
</style>
""", unsafe_allow_html=True)


background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
  background-image: url("https://images.unsplash.com/photo-1604147706283-d7119b5b822c?q=80&w=2959&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
  background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
  background-position: center;  
  background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)


with st.sidebar:
    st.title("Google Play Store Reviews")
    choice = st.radio("Navigation", ["Dense Retrieval", "Hybrid Retrieval"], index=0)
    st.markdown("This application allows you to retrieve most relevant reviews based on your query", unsafe_allow_html=True)

pc = Pinecone(api_key="1ccf9653-ea31-483b-bbb5-bd811dd25073")

def create_card(user_name, review_date, rating, review):
    # Define color based on rating
    if rating == 1.0:
        color = '#FF0000' # Red
    elif rating == 2.0:
        color = '#FF5733' # Orange
    elif rating == 3.0:
        color = '#FFC300' # Yellow
    elif rating == 4.0:
        color = '#00AAB5' # Chartreuse
    else: # rating == 5
        color = '#1B5400' # Green

    card = f"""
    <div style="
        border:2px solid black; 
        border-radius:15px; 
        padding:10px; 
        margin:10px 0;
        width:100%;
        background-color:#E2E2E2;">
        <div style="
            display:flex; 
            justify-content:space-between;">
            <div style="
                border:1px solid black;
                border-radius:5px;
                padding:5px;
                font-size: 14px; 
                font-weight: bold;
                background-color:white;">
                {user_name}
            </div>
            <div style="
                border:1px solid black;
                border-radius:5px;
                padding:5px;
                font-size: 14px; 
                font-weight: bold;
                background-color:white;">
                {review_date}
            </div>
            <div style="
                border:1px solid black;
                border-radius:5px;
                padding:5px;
                font-size: 14px; 
                font-weight: bold;
                text-align: right;
                color:{color};
                background-color:white;">
                Rating: {rating}
            </div>
        </div>
        <div style="
            border:1px solid black;
            border-radius:15px;
            padding:10px;
            height:100px; 
            overflow:auto; 
            margin-top:10px;
            background-color:white;">
            {review}
        </div>
    </div>
    """
    return card






def dataframe_to_html_with_border(df):



    # Converting DataFrame to HTML and add a border, width, and bottom margin
    df_html = df.to_html(index=False, escape=False).replace('<table', '<table style="border:2px solid black; width:100%; margin-bottom:20px;"')

    # Wrapping 'brief_summary' and 'eligibility' data in a scrollable div
    df_html = df_html.replace('<td>', '<td><div style="width: 150px;height : 150px; overflow-x: auto;">')
    df_html = df_html.replace('</td>', '</div></td>')

    return df_html

@st.cache_resource()
def load_model(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('Sorry no cuda.')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model


if choice == 'Dense Retrieval' :

  if 'query' not in st.session_state:
        st.session_state['query'] = 'Which reviews are talking about coins?'

  if 'num_results' not in st.session_state:
        st.session_state['num_results'] = 5
  index = pc.Index('hybrid-index')
  model = load_model()
  query = st.text_input('What do you wanna search?', value=st.session_state['query'], key='query_sparse')
  num_results = st.slider('Number of reviews', 1, 100, st.session_state['num_results'])

  def run_query(query):
    dense = model.encode(query).tolist()
    results = index.query(
    top_k=num_results,
    vector = dense,
    include_metadata=True)
    return results
  
  
  if st.button('Get Results'):

    result = run_query(query)

    matches = result['matches']

    # Extract the 'metadata' dictionaries from each item in matches
    metadata = [match['metadata'] for match in matches]

    # Create a DataFrame from the list of metadata dictionaries
    df = pd.DataFrame(metadata)
    df = df.drop(columns=['Unnamed: 0'])
    df.columns = ['review_date', 'review', 'rating', 'thumbsUpCount', 'userName']
    # Print the DataFrame
    # df_html = dataframe_to_html_with_border(df)
    # st.write(df_html, unsafe_allow_html=True)

    # Iterate over the DataFrame rows and create a card for each row
    for _, row in df.iterrows():
        card = create_card(row['userName'], row['review_date'], row['rating'], row['review'])
        st.markdown(card, unsafe_allow_html=True)
        
        








if choice == 'Hybrid Retrieval' :

  if 'query' not in st.session_state:
        st.session_state['query'] = 'Which reviews are talking about coins?'

  if 'num_results' not in st.session_state:
        st.session_state['num_results'] = 5
  index = pc.Index('hybrid-index')
  model = load_model()
  query = st.text_input('What do you wanna search?', value=st.session_state['query'], key='query_sparse')
  num_results = st.slider('Number of reviews', 1, 100, st.session_state['num_results'])

  def run_query(query):
    bm25 = BM25Encoder().default()
    dense = model.encode(query).tolist()
    sparse = bm25.encode_queries(query)

    results = index.query(
    sparse_vector=sparse,
    top_k=num_results,
    vector = dense,
    include_metadata=True)
    return results
  
  if st.button('Get Results'):
      
    result = run_query(query)

    matches = result['matches']

    # Extract the 'metadata' dictionaries from each item in matches
    metadata = [match['metadata'] for match in matches]

    # Create a DataFrame from the list of metadata dictionaries
    df = pd.DataFrame(metadata)
    df = df.drop(columns=['Unnamed: 0'])
    df.columns = ['review_date', 'review', 'rating', 'thumbsUpCount', 'userName']
    # Print the DataFrame
    # df_html = dataframe_to_html_with_border(df)
    # st.write(df_html, unsafe_allow_html=True)

        # Iterate over the DataFrame rows and create a card for each row
    for _, row in df.iterrows():
        card = create_card(row['userName'], row['review_date'], row['rating'], row['review'])
        st.markdown(card, unsafe_allow_html=True)








   
  


