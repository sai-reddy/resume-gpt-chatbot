import streamlit as st

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.components.v1 import html

import os
from io import StringIO

CHUNK_SIZE = 2000
PINECONE_INDEX_NAME = 'ai-doc-qa'


def init_pinecone():
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):   
    # Create llm and chain to answer questions from pinecone index
    llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = load_qa_chain(llm, chain_type="stuff")

    if prompt:        
        docs = st.session_state['pinecone_index'].similarity_search(prompt)
        response = chain.run(input_documents=docs, question=prompt)
    return response  
 
def index_resume():
    # doc =  './data/sai_kalyanreddy_pentaparthi_resume.pdf'
    # loader = UnstructuredPDFLoader(doc)
    loader = OnlinePDFLoader('https://storage.googleapis.com/resume-gpt-chatbot/sai_kalyanreddy_pentaparthi_resume.pdf')
    data = loader.load()

    # Split into smallest docs possible
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # If vector count is nearing free limits delete index and recreate it
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():        
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric='cosine')

    # Create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])  
    st.session_state['pinecone_index'] = Pinecone.from_texts([t.page_content for t in texts],
                                                                embeddings, index_name=PINECONE_INDEX_NAME)
@st.cache_resource
def waking_up_bot():
    init_pinecone()
    index_resume()
    st.success('Bot is Ready')
    
# App framework
def app():
    st.set_page_config(page_title="Sai Resume ChatBot - An OPENAI LLM-powered Resume Chat App for Sai", page_icon=":robot:")
    waking_up_bot()

    with st.sidebar:
        st.title('🦜️🔗 SAI RESUME GPT CHATBOT')
        st.markdown('''
        ## About
        This app is an OPENAI LLM-powered Resume Chatbot for Sai
        ''')
        add_vertical_space(5)
        st.write('Made By Sai Pentaparthi(saikalyanr.p@gmail.com)')
        html(f'''
            <a href="www.linkedin.com/in/saikalyanrp" style="color:#ffffff;">LinkedIn Profile</a>
             ''')

    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi!, I'm Sai Resume GPT, What do you want to know about Sai?"]
    ## past stores User's questions
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Howdy!']

    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

   ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def cleanup():
    pass

if __name__ == '__main__':    
    try:
        app()
    finally:
        cleanup()