import streamlit as st
import berserk
import chess
import time
from db import execute_db_query
from rag.orchestrator import generate_fen_embeddings
from neo4j import Result
from neo4j_viz import neo4j
from util import load_file
from models import ChessGame, ChessMove, MoveAnalysisJudgement, Side, ChessOpening

insert_query = load_file("queries/insert_games.cypher")
fetch_query = load_file("queries/fetch_games.cypher")

@st.cache_resource
def get_lichess_client():
	li_session = berserk.TokenSession(st.secrets.LICHESS_API_TOKEN)
	return berserk.Client(session=li_session)

li_client = get_lichess_client()
board = chess.Board()

@st.cache_data(ttl=86400)
def fetch_games_for_user(username):
  games_iterator = li_client.games.export_by_player(
    username=username,
    perf_type="blitz,rapid",
    max=st.secrets.GAME_FETCH_COUNT,
    as_pgn=False,
    moves=True,
    evals=True,
    analysed=True,
    opening=True
  )
  return [x for x in games_iterator] 

def get_player_id(game, side):
  if 'players' in game and side in game['players'] and 'user' in game['players'][side] and 'id' in game['players'][side]['user']:
    return str.lower(game['players'][side]['user']['id'])
  return None

def get_move_judgement(move_analysis):
  # print(move_analysis)
  eval = "NotAvailable"
  judgement = MoveAnalysisJudgement.NONE
  if not move_analysis:
    return eval, judgement, False
  
  is_forced_mate = 'mate' in move_analysis
  if is_forced_mate:
    eval = move_analysis['mate']
    judgement = MoveAnalysisJudgement.MATE
  else:
    eval = move_analysis['eval'] if 'eval' in move_analysis else "NotAvailable"
    judgement = MoveAnalysisJudgement(move_analysis['judgment']['name']) if 'judgment' in move_analysis and 'name' in move_analysis['judgment'] else MoveAnalysisJudgement.NONE
    # print(judgement)
  return eval, judgement, is_forced_mate
def process_game_v2(game):
  print("Processing game: ", game)
  board.reset()
  game_id = game['id']
  white_id = get_player_id(game, 'white')
  black_id = get_player_id(game, 'black')
  status = game['status']
  if not white_id:
    white_id = f"unknown_player_white_{game_id}"
  if not black_id:
    black_id = f"unknown_player_white_{game_id}"
  ch = ChessGame(
    game_id=game_id, 
    white_id=white_id, 
    black_id=black_id,
    status=status,
    game_speed=game['perf'],
    played_on=game['createdAt'],
    opening=ChessOpening(
      name=game['opening']['name'] if 'opening' in game and 'name' in game['opening']   else "NotAvailable",
      eco=game['opening']['eco'] if 'opening' in game and 'eco' in game['opening'] else "NotAvailable"
    ),
    winner_id=(white_id if game['winner'] == "white" else black_id) if status not in ["draw", "stalemate"] else "NotApplicable",
    winning_side=(Side.WHITE if game['winner'] == "white" else Side.BLACK) if status not in ["draw", "stalemate"] else "NotApplicable"
  )
  if "moves" in game and len(game['moves']) > 0:
    chess_moves = []
    ls_san_moves = game['moves'].split(" ")
    # print("Number of moves: ", len(ls_san_moves))
    # print("Number of analysis objs: ", len(game['analysis']))
    for i in range(0, len(ls_san_moves)):
      san = ls_san_moves[i]
      move_analysis = game['analysis'][i] if i < len(game['analysis']) else None
      # Get FEN
      board.push_san(san)
      fen = board.fen()
      # Get Analysis Info
      eval, judgement_name, is_forced_mate = get_move_judgement(move_analysis)
      cm = ChessMove(
        move_number=i+1,
        moving_side=Side.WHITE if i % 2 == 0 else Side.BLACK,
        move_san=san,
        move_fen=fen,
        analysis_type=judgement_name,
        forced_mate=is_forced_mate,
        move_eval=eval
      )
      chess_moves.append(cm)
    ch.moves = chess_moves
    ch.n_moves = len(chess_moves) // 2 if len(chess_moves) % 2 == 0 else len(chess_moves) // 2 + 1
  print("Processed Game:", ch)
  return ch

# def process_game(game):
#   # print(f"Entering process_games. Processing game {game['id']}")
#   board.reset()
#   if not 'user' in game['players']['white']:
#     # print(f"User not found: {game}")
#     player_id = f"unknown_player_black_{game['id']}"
#     game['players']['white']['user'] = dict()
#     game['players']['white']['user']['id'] = player_id
#     game['players']['white']['user']['name'] = player_id
#   if not 'user' in game['players']['black']:
#     # print(f"User not found: {game}")
#     player_id = f"unknown_player_white_{game['id']}"
#     game['players']['black']['user'] = dict()
#     game['players']['black']['user']['id'] = player_id
#     game['players']['black']['user']['name'] = player_id
#   print(f"Game: {game}")
#   if 'moves' in game and len(game['moves']) > 0:
#     moves = game['moves']
#     ls_san_moves = moves.split(" ")
#     ls_final_moves = []
#     for san in ls_san_moves:
#       board.push_san(san)
#       fen = board.fen()
#       ls_final_moves.append(dict(fen=fen, san=san))
#       game['moves_final']=ls_final_moves
#   return game

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
  games_processed = [process_game_v2(game) for game in games]
  print("Games processed", games_processed)
  
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
  
  st.write("Games loaded successfully")
  t2 = time.time()
  print(f"Loaded games in {t2-t1} seconds")
  return vg