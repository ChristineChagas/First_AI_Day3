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
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")


st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
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
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title('News Summarizer Tool')
   st.write("Welcome to the News Article Summarizer Tool, designed to provide you with clear, concise, and well-structured summaries of news articles. This tool is ideal for readers who want to quickly grasp the essential points of any news story without wading through lengthy articles. Whether you’re catching up on global events, diving into business updates, or following the latest political developments, this summarizer delivers all the important details in a brief, easily digestible format.")
   st.write("## What the Tool Does")
   st.write("The News Article Summarizer Tool reads and analyzes full-length news articles, extracting the most critical information and presenting it in a structured manner. It condenses lengthy pieces into concise summaries while maintaining the integrity of the original content. This enables users to quickly understand the essence of any news story.")
   st.write("## How It Works")
   st.write("The tool follows a comprehensive step-by-step process to create accurate and objective summaries:")
   st.write("*Analyze and Extract Information:* The tool carefully scans the article, identifying key elements such as the main event or issue, people involved, dates, locations, and any supporting evidence like quotes or statistics.")
   st.write("*Structure the Summary:* It organizes the extracted information into a clear, consistent format. This includes:")
   st.write("- *Headline:* A brief, engaging headline that captures the essence of the story.")
   st.write("- *Lead:* A short introduction summarizing the main event.")
   st.write("- *Significance:* An explanation of why the news matters.")
   st.write("- *Details:* A concise breakdown of the key points.")
   st.write("- *Conclusion:* A wrap-up sentence outlining future implications or developments.")
   st.write("# Why Use This Tool?")
   st.write("- *Time-Saving:* Quickly grasp the key points of any article without having to read through long pieces.")
   st.write("- *Objective and Neutral:* The tool maintains an unbiased perspective, presenting only factual information.")
   st.write("- *Structured and Consistent:* With its organized format, users can easily find the most relevant information, ensuring a comprehensive understanding of the topic at hand.")
   st.write("# Ideal Users")
   st.write("This tool is perfect for:")
   st.write("- Busy professionals who need to stay informed but have limited time.")
   st.write("- Students and researchers looking for quick, accurate summaries of current events.")
   st.write("- Media outlets that want to provide readers with quick takes on trending news.")
   st.write("Start using the News Article Summarizer Tool today to get concise and accurate insights into the news that matters most!")
   
elif options == "About Us" :
     st.title('News Summarizer Tool')
     st.subheader("About Us")
     st.write("# Danielle Bagaforo Meer")
     st.image('images/Meer.png')
     st.write("## AI First Bootcamp Instructor")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/algorexph/")
     st.text("Kaggle Account : https://www.kaggle.com/daniellebagaforomeer")
     st.write("\n")


elif options == "Model" :
     st.title('News Summarizer Tool')
     col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
          News_Article = st.text_input("News Article", placeholder="News : ")
          submit_button = st.button("Generate Summary")

     if submit_button:
        with st.spinner("Generating Summary"):
             System_Prompt = """
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
             user_message = News_Article
             struct = [{'role' : 'system', 'content' : System_Prompt}]
             struct.append({"role": "user", "content": user_message})
             chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
             response = chat.choices[0].message.content
             struct.append({"role": "assistant", "content": response})
             st.success("Insight generated successfully!")
             st.subheader("Summary : ")
             st.write(response)
