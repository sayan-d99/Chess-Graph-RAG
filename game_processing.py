import streamlit as st
import berserk
import chess
import time
from db import execute_db_query
from rag import generate_fen_embeddings
from neo4j import Result
from neo4j_viz import neo4j

def load_query(filename):
  with open(filename, 'r') as f:
    return f.read()

insert_query = load_query("queries/insert_games.cypher")
fetch_query = load_query("queries/fetch_games.cypher")

@st.cache_resource
def get_lichess_client():
	li_session = berserk.TokenSession(st.secrets.LICHESS_API_TOKEN)
	return berserk.Client(session=li_session)

li_client = get_lichess_client()
board = chess.Board()

@st.cache_data(ttl=86400)
def fetch_games_for_user(username):
  # return [li_client.games.export('pWl6FEXl', as_pgn=False, moves=True, opening=True, literate=False, clocks=False)]
  games_iterator = li_client.games.export_by_player(
    username=username,
    perf_type="blitz,rapid",
    max=st.secrets.GAME_FETCH_COUNT,
    opening=True,
    literate=True,
  )
  return [x for x in games_iterator] 

def process_game(game):
  # print(f"Entering process_games. Processing game {game['id']}")
  board.reset()
  if not 'user' in game['players']['white']:
    # print(f"User not found: {game}")
    player_id = f"unknown_player_black_{game['id']}"
    game['players']['white']['user'] = dict()
    game['players']['white']['user']['id'] = player_id
    game['players']['white']['user']['name'] = player_id

  if not 'user' in game['players']['black']:
    # print(f"User not found: {game}")
    player_id = f"unknown_player_white_{game['id']}"
    game['players']['black']['user'] = dict()
    game['players']['black']['user']['id'] = player_id
    game['players']['black']['user']['name'] = player_id

  # print(f"Game: {game}")

  if 'moves' in game and len(game['moves']) > 0:
    moves = game['moves']
    ls_san_moves = moves.split(" ")
    ls_final_moves = []
    for san in ls_san_moves:
      board.push_san(san)
      fen = board.fen()
      ls_final_moves.append(dict(fen=fen, san=san))
      game['moves_final']=ls_final_moves

  return game

def load_games(li_username):
  t1 = time.time()
  st.write(f"Fetching lichess games of {li_username}")
  
  print(f"Fetching lichess games for {li_username}")
  games = fetch_games_for_user(li_username)
  # print(f"Games Fetched: {games}")
  print(f"Fetched {len(games)} for user {li_username}")
  
  st.write(f"Fetched {len(games)} games")
  st.write("Processing Games")
  
  print("Processing games")
  games_processed = [process_game(game) for game in games]
  print("Games processed")
  
  st.write("Games processed")
  st.write("Saving Games")

  print("Inserting games in neo4j")
  insert_time_1 = time.time()
  insert_result = execute_db_query(insert_query, params=dict(games=games_processed))
  insert_time_2 = time.time()
  print(f"Games Inserted {insert_result}")
  print(f"Games Inserted in {insert_time_2 - insert_time_1} seconds. Insertion Summary: {insert_result.summary.counters}")
  
  print("Fetching game data for visualization")
  game_data_graph = execute_db_query(fetch_query, params=dict(playerId=li_username), result_transformer=Result.graph)
  print(f"Data Fetched. Received transformed graph object")
  
  fen_time_1 = time.time()
  print("Generate FEN embedding")
  generate_fen_embeddings()
  fen_time_2 = time.time()
  print(f"FEN embeddings generated in {fen_time_2 - fen_time_1} seconds")
  
  print("Building Visualization graph")
  vg = neo4j.from_neo4j(game_data_graph)
  print(f"Visualization graph built with {len(vg.nodes)} nodes and {len(vg.relationships)} relations")
  
  st.write("Games saved")
  t2 = time.time()
  print(f"Loaded games in {t2-t1} seconds")
  return vg