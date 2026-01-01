import streamlit as st
import berserk
import chess

@st.cache_resource
def get_lichess_client():
	li_session = berserk.TokenSession(st.secrets.LICHESS_API_TOKEN)
	return berserk.Client(session=li_session)

li_client = get_lichess_client()
board = chess.Board()

def get_game_players(game):
  player_white = game['players']['white']["user"]
  player_black = game['players']['black']["user"]
  return player_white, player_black

@st.cache_data(ttl=86400)
def fetch_games_for_user(username):
  print(f"Entering fetch_games_for_user for {username}")
  games_iterator = li_client.games.export_by_player(
    username=username,
    perf_type="blitz,rapid",
    max=st.secrets.GAME_FETCH_COUNT,
    opening=True,
    literate=True,
    tags=True
  )
  return [x for x in games_iterator] 

def process_games(game):
  # print(f"Entering process_games. Processing game {game['id']}")
  board.reset()
  if 'moves' in game and len(game['moves']) > 0:
    moves = game['moves']
    ls_san_moves = moves.split(" ")
    ls_final_moves = []
    for san in ls_san_moves:
      board.push_san(san)
      fen = board.fen()
      # cp = fetch_fen_analysis(fen)
      ls_final_moves.append(dict(fen=fen, san=san))
      game['moves_final']=ls_final_moves
  return game