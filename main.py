import streamlit as st
import streamlit.components.v1 as components
import time
import random
import logging

from game_processing import load_games
from rag.orchestrator import execute_rag_query
from models import ChatMessage
from db.db import execute_db_query

def setup_session():
  keys = [
    ("messages", []), 
    ("username", ""), 
    ("is_processing_prompt", False), 
    ("current_prompt", ""), ("vg", None),
    ("game_data_loaded", False)
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
st.set_page_config(page_title="Chess GPT", layout="wide")
st.title("Chess GPT")

def set_current_user_input(input_str: str):
  st.session_state.current_prompt = input_str  
  st.session_state.is_processing_prompt = True
  st.rerun(scope="fragment")

col1, col2 = st.columns(2)

with col1:
  player_input = st.text_input("Enter Lichess User Name", placeholder="Enter Lichess User Name")
  st.session_state.username = player_input
  if st.button("Load My Games", disabled=len(player_input) == 0):
    st.session_state.messages = []
    with st.status(f"Loading games of {player_input}") as status: 
      st.session_state.vg = load_games(player_input)
      st.session_state.game_data_loaded = True
      status.update(label="Games have been loaded. Fire your queries", expanded=False)
    with st.container(border=True):
      html_content = st.session_state.vg.render()
      components.html(html_content.data, height=400, scrolling=True)

@st.fragment
def render_chat():
  chat_container = st.container(height=400)
  with chat_container:
    for msg in st.session_state.messages:
      msg.render(st)
  
  user_prompt = st.chat_input(
    "What do you want to know about your games?" if st.session_state.game_data_loaded else "Load your games before you can ask a question",
    disabled=not st.session_state.game_data_loaded or st.session_state.is_processing_prompt
  )
  if user_prompt:
    set_current_user_input(user_prompt)
    
  if st.session_state.game_data_loaded and st.session_state.is_processing_prompt and st.session_state.current_prompt:
    try:
      user_chat_message = ChatMessage(role="human",text=st.session_state.current_prompt,has_chart=False,chart_data=None)  
      st.session_state.messages.append(user_chat_message)
      
      with chat_container:
        user_chat_message.render(st)
        
        ai_loading_chat_message = ChatMessage(role="ai", text="Thinking....",has_chart=False,chart_data=None)
        st.session_state.messages.append(ai_loading_chat_message)
        ai_loading_chat_message.render(st)
        
        # st.rerun(scope="fragment")
        response = execute_rag_query(
          st.session_state.current_prompt, 
          st.session_state.username.lower(), 
          st.session_state.messages[:-2]
        )
        print(f"Query: {st.session_state.current_prompt} Response: {response}")
        
        ai_chat_message = ChatMessage.from_llm_response(response)
        ai_chat_message.render(st)
        
        st.session_state.messages.pop()
        st.session_state.messages.append(ai_chat_message)
        
        try:
          # params = dict()
          # params['pid'] = st.session_state.username
          # params['query'] = st.session_state.current_prompt
          # params['gdb_query'] = response.intermediate_step
          # params['response'] = response.response
          # execute_db_query("CREATE Query(username: $pid, query: $query, response: $res)", params)
          pass
        except Exception as e:
          print(e)
    
    except Exception as e:
      st.toast("Something went wrong. Please try again")
    finally:
      st.session_state.is_processing_prompt = False
      st.session_state.current_prompt = None
      st.rerun(scope="fragment")

with col2:
  render_chat()
