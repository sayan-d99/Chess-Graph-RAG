import streamlit as st
import streamlit.components.v1 as components

from game_processing import load_games

# from rag import execute_rag_query

def setup_session():
  if "queries" not in st.session_state:
    st.session_state.queries = []  
  if "vg" not in st.session_state:
    st.session_state.vg = None

setup_session()
st.set_page_config(page_title="NeoChess", layout="wide")
st.title("Welcome to NeoChess")
col1, col2 = st.columns(2)
# player_input = ""

with col1:
  player_input = st.text_input(
    "Enter Lichess Username", placeholder="Enter Lichess User Name"
  )
  if st.button("Visualize Graph", disabled=len(player_input) == 0):
    with st.status("Loading games of " + player_input) as status:
      st.session_state.vg = load_games(player_input)
      status.update(label="Data Loading complete", expanded=False)
    with st.container(border=True):
      html_content = st.session_state.vg.render()
      components.html(html_content.data, height=400, scrolling=True)

@st.fragment
def render_chat():
  # Initialize queries if not present
  for q in st.session_state.queries:
    st.chat_message(q["role"]).markdown(q["content"])
  if prompt := st.chat_input("Ask about your games"):
    # st.chat_message("user").markdown(prompt)
    st.session_state.queries.append({"role": "user", "content": prompt})
    response = f"Echo: {prompt}"
    # st.chat_message("ai").markdown(response)
    st.session_state.queries.append({"role": "ai", "content": response})
    st.rerun(scope="fragment")

with col2:
  render_chat()


# with col2:
#   for q in st.session_state.queries:
#     st.chat_message(q['role']).markdown(q['content'])

#   if prompt := st.chat_input("Ask a question about your Chess games"):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.queries.append({"role": "user", "content": prompt})
#     response = f"Echo {prompt}"
#     st.chat_message("ai").markdown(response)
#     st.session_state.queries.append({"role": "ai", "content": response})
