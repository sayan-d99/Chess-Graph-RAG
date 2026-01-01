import streamlit as st
import streamlit.components.v1 as components
from neo4j_viz import VisualizationGraph
from neo4j_viz.neo4j import from_neo4j
import time 

from db_utils import insert_games
from chess_utils import fetch_games_for_user, process_games
from rag import get_query_response

def load_games(li_username):
  t1 = time.time()
  st.write("Fetching games of the last week from Lichess")
  games = fetch_games_for_user(li_username)
  print(games)
  st.write(f"Fetched {len(games)} games")
  st.write("Processing Games")
  games_processed = [process_games(game) for game in games]
  st.write("Games processed")
  st.write("Create knowledge Graph")
  insert_games(games_processed)
  st.write("Knowledge graph ready")
  t2 = time.time()
  print(f"Loaded games in {t2-t1} seconds")

st.title("Welcome to Lichess-Neo4j")

player_input = st.text_input("Enter Lichess Username", value="HoozYourDaddy")

if "queries" not in st.session_state:
  st.session_state.queries = []

if st.button("Visualize Graph"):
  with st.status("Loading games of " + player_input) as status:
    load_games(player_input)
    status.update(label="Data Loading complete", expanded=False)
    
  if query_prompt := st.chat_input("Ask a question about your games?"):
    st.session_state.queries.append({"role": "user", "content": query_prompt})
    
    with st.chat_message("user"):
      st.markdown(query_prompt)
    
    for message in st.session_state.queries:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    response = get_query_response(query_prompt)
    st.session_state.queries.append({"role": "ai", "content": query_prompt})
    
    with st.chat_message("ai"):
      st.markdown(response)
    
  # with st.spinner("Fetching graph data..."):
  #   try:
  #     # Fetch the nodes and relationships
  #     viz_data = get_graph_data(player_input)
      
  #     # Render to HTML
  #     # We use height='600px' to ensure it fits the Streamlit container
  #     html_content = viz_data.render()
      
  #     # Display in Streamlit using the components' HTML renderer
  #     components.html(
  #         html_content.data,
  #         height=600,
  #         scrolling=True)
  #   except Exception as e:
  #         st.error(f"Error fetching data: {e}")

# vgGraph = get_graph_data(player_input)
# nodes = vgGraph.nodes
# relationships = vgGraph.relationships

# for node in nodes:
#     if "Player" in node.:
#         node.color = "#FF4B4B"  # Red for players
#         node.size = 30
#     elif "FEN" in node.labels:
#         node.color = "#1C83E1"  # Blue for positions
#         node.size = 15

# vg = VisualizationGraph(nodes, relationships)
