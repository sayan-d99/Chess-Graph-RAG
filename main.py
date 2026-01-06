import streamlit as st
import streamlit.components.v1 as components
import time
import random

from game_processing import load_games
from rag import execute_rag_query

def setup_session():
  keys = [
    ("queries", []), 
    ("username", ""), 
    ("is_processing_prompt", False), 
    ("current_prompt", ""), ("vg", None)
  ]
  for k in keys:
    key = k[0]
    value = k[1]
    if key not in st.session_state:
      st.session_state[key] = value

def get_chat_message(role, message):
  return {"role": role, "content": message}

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
  player_input = st.text_input(
    "Enter Lichess Username", placeholder="Enter Lichess User Name"
  )
  st.session_state.username = player_input
  if st.button("Visualize Graph", disabled=len(player_input) == 0):
    with st.status("Loading games of " + player_input) as status:
      st.session_state.vg = load_games(player_input)
      status.update(label="Data Loading complete", expanded=False)
    with st.container(border=True):
      html_content = st.session_state.vg.render()
      components.html(html_content.data, height=400, scrolling=True)
  
@st.fragment
def render_chat():
  print(f"Messages: {st.session_state.queries}")
  chat_container = st.container(height=400)
  
  with chat_container:
    for q in st.session_state.queries:
      st.chat_message(q["role"]).markdown(q["content"])
  
  user_prompt = st.chat_input(
    "What do you want to know about your games?",
    disabled=st.session_state.is_processing_prompt
  )
  
  if user_prompt:
    st.session_state.processing = True  
    st.session_state.current_prompt = user_prompt  
    st.session_state.is_processing_prompt = True
    st.rerun(scope="fragment")
    
  if st.session_state.is_processing_prompt and st.session_state.current_prompt:  
    st.session_state.queries.append(get_chat_message("user", st.session_state.current_prompt))
    response = "Something went wrong when answering the query"
    with chat_container:
      st.chat_message("user").markdown(st.session_state.current_prompt)
      response = execute_rag_query(st.session_state.current_prompt, st.session_state.username.lower())
      # print('AI response:', response)
      st.chat_message("assistant").markdown(response)
      st.session_state.queries.append(get_chat_message("assistant", response))
    
    st.session_state.is_processing_prompt = False
    st.session_state.current_prompt = None
    st.rerun(scope="fragment")

with col2:
  render_chat()
