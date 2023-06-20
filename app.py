import os

import pinecone
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from streamlit.components.v1 import html
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

CHUNK_SIZE = 2000
PINECONE_INDEX_NAME = 'resume-gpt-chatbot'


def init_pinecone():
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


# Response output
# Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    response = None
    # Create llm and chain to answer questions from pinecone index
    llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = load_qa_chain(llm, chain_type="stuff")

    if prompt:
        docs = st.session_state['pinecone_index'].similarity_search(prompt)
        response = chain.run(input_documents=docs, question=prompt)
    return response


def get_pinecone_index():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    st.session_state['pinecone_index'] = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME,
                                                                      embedding=embeddings)


def waking_up_bot():
    if st.session_state.get('pinecone_index') is None:
        with st.spinner('Waking up bot'):
            init_pinecone()
            get_pinecone_index()
        st.success('Bot is Ready')


# App framework
def app():
    st.set_page_config(page_title="Sai Resume ChatBot - An OPENAI ChatGPT LLM-powered", page_icon=":robot:")
    waking_up_bot()

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    with st.sidebar:
        st.title('🦜️🔗 SAI RESUME ChatGPT')
        st.markdown('''
        ## About
        Chatbot powered by OPENAI ChatGPT LLM models
        ''')
        add_vertical_space(5)
        st.write('Made By Sai Pentaparthi(saikalyanr.p@gmail.com)')
        html(f'''
            <a href="www.linkedin.com/in/saikalyanrp" style="color:#ffffff;">LinkedIn Profile</a>
             ''')

    # Generate empty lists for generated and past.
    # generated stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["""Hi!, I'm Sai Resume ChatGPT, You can ask me questions like
                                         Who is sai
                                         Where is sai working currently
                                         Where did he graduate from
                                         what databases he know
                                         what did sai do with postgresql
                                         etc..."""]
    # past stores User's questions
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Howdy!']

    response_container = st.container()
    colored_header(label='', description='', color_name='blue-70')
    input_container = st.container()

    # Applying the user input box
    with input_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=50)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

    # Conditional display of AI generated responses as a function of user provided prompts
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))


def cleanup():
    pass


if __name__ == '__main__':
    try:
        app()
    finally:
        cleanup()
