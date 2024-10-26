import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not (openai_api_key.startswith("sk-") and len(openai_api_key) == 51) :
        st.warning("Please enter a valid OpenAI API key!")
    else :
        st.success("API key valid!")

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin"}})

if 'messages' not in st.session_state :
    st.session_state.messages = []

if 'chat_session' not in st.session_state :
    st.session_state.chat_session = None

elif options == "Home" :
    st.title("News Summarizer Tool")
    st.write("This is a tool that summarizes news articles.")

elif options == "About Us" :
    st.title("About Us")
    st.write("This is a tool that summarizes news articles.")

elif options == "Model" :
    st.title("News Summarizer Tool")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2 :
        News_Article = st.text_input("Enter News Article", placeholder = "Enter News Article Here...")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary..."):
            System_Prompt = System_Prompt =

 """
A Groundbreaking Discovery
You’re in for an exciting journey into the underwater world of sperm whales! Researchers at MIT have achieved a remarkable breakthrough by revealing a 'phonetic alphabet' hidden within the whales' clicks. This study employs advanced artificial intelligence to analyze nearly 9,000 recordings from a specific clan of sperm whales in the Eastern Caribbean.

Understanding Whale Communication
What’s particularly intriguing about this research is the identification of 156 distinct click patterns—a significant increase from the previously accepted number of 21. These click sequences operate much like phonemes in human languages, suggesting that sperm whales possess a sophisticated system for crafting intricate messages.

The Role of AI in Analysis
The AI analysis didn’t just stop at identifying different clicks; it also picked up on subtle variations in rhythm and tempo. These findings imply that whale communication can carry contextual meanings, much like how we use tone and pacing in our speech. This nuanced understanding opens new avenues for exploring the social interactions and decision-making processes among these intelligent marine mammals.

Linguistic Parallels
Moreover, this groundbreaking study invites you to ponder the parallels between whale communication and human language, encouraging deeper questions about how different species convey information. The implications of these findings could reshape our understanding of animal communication and its complexity.

Dive Deeper
If you’re eager to dive deeper into this fascinating study, check out the full article on BBC Future
NEURONAD
ps://neuronad.com/ai-news/science/the-sperm-whale-phonetic-alphabet-revealed-by-ai/). You'll discover even more insights into the remarkable world of sperm whales and their intricate communication systems.
"""

            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role" : "assistant", "content" : response})
            print("Assistant", response)
