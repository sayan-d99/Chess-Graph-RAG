import streamlit as st
import streamlit.components.v1 as components
import time
import random
import pandas as pd

from game_processing import load_games
from rag.orchestrator import execute_rag_query
from mytypes import ChatMessage

def setup_session():
  keys = [
    ("queries", []), 
    ("username", ""), 
    ("is_processing_prompt", False), 
    ("current_prompt", ""), ("vg", None),
    ("game_data_loaded", False),
    ("suggestions", [])
  ]
  for k in keys:
    key = k[0]
    value = k[1]
    if key not in st.session_state:
      st.session_state[key] = value


def generate_response():
  response = "This is a response.This is a response.This is a response.This is a response"
  for word in response.split():
    yield word + " "
    time.sleep(random.randint(1, 5) * 0.1)

# UI CODE BEGINS HERE

setup_session()
st.set_page_config(page_title="ChessRAG", layout="wide")
st.title("ChessRAG")
col1, col2 = st.columns(2)

with col1:
  player_input = st.text_input("Enter Lichess User Name", placeholder="Enter Lichess User Name")
  st.session_state.username = player_input
  if st.button("Load My Games", disabled=len(player_input) == 0):
    with st.status(f"Loading games of {player_input}") as status:
      st.session_state.vg = load_games(player_input)
      st.session_state.game_data_loaded = True
      status.update(label="Games have been loaded. Fire your queries", expanded=False)
    with st.container(border=True):
      html_content = st.session_state.vg.render()
      components.html(html_content.data, height=600, scrolling=True)

def set_current_user_input(input_str):
  st.session_state.suggestions = []
  st.session_state.processing = True  
  st.session_state.current_prompt = input_str  
  st.session_state.is_processing_prompt = True
  st.rerun(scope="fragment")

@st.fragment
def render_chat():
  print(f"Messages: {st.session_state.queries}")
  chat_container = st.container(height=600)
  with chat_container:
    for msg in st.session_state.queries:
      msg.render(st)
  
  if st.session_state.suggestions and len(st.session_state.suggestions) > 0:
    selection = st.pills("Suggestions", st.session_state.suggestions)
    set_current_user_input(selection)
      
  user_prompt = st.chat_input(
    "What do you want to know about your games?" if st.session_state.game_data_loaded else "Load your games before you can ask a question",
    disabled=not st.session_state.game_data_loaded or st.session_state.is_processing_prompt
  )
  if user_prompt:
    set_current_user_input(user_prompt)
    
  if st.session_state.game_data_loaded and st.session_state.is_processing_prompt and st.session_state.current_prompt:
    user_chat_message = ChatMessage(
      role="human",
      text=st.session_state.current_prompt,
      has_chart=False,
      chart_df=None,
      chart_type=None
    )  
    st.session_state.queries.append(user_chat_message)
    
    with chat_container:
      user_chat_message.render(st)
      response = execute_rag_query(st.session_state.current_prompt, st.session_state.username.lower())
      ai_chat_message = ChatMessage.from_llm_response(response)
      ai_chat_message.render(st)
      st.session_state.queries.append(ai_chat_message)
      st.session_state.suggestions = response.suggestions
    
    st.session_state.is_processing_prompt = False
    st.session_state.current_prompt = None
    st.rerun(scope="fragment")

with col2:
  render_chat()
