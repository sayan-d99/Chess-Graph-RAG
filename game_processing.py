import streamlit as st
import berserk
import chess
import time
from db import execute_db_query
from neo4j import Result
from neo4j_viz import neo4j

GAMES_INSERT_QUERY = """
	UNWIND $games AS game
  // 1. Process Players and Openings
  MERGE (w:Player {pid: game.players.white.user.id})
  ON CREATE SET w.name = game.players.white.user.name

  MERGE (b:Player {pid: game.players.black.user.id})
  ON CREATE SET b.name = game.players.black.user.name 

  MERGE (o:Opening {eco: game.opening.eco})
  ON CREATE SET o.name = game.opening.name

  // 2. Create the Game
  MERGE (g:Game {id: game.id})
  SET g.gameType = game.perf,
      g.status = game.status,
      g.winningSide = coalesce(game.winner, "NO_WINNER"),
      g.fullId = game.fullId,
      g.playedOn = game.createdAt,
      g.winnerId = coalesce(
          CASE 
              WHEN game.winner = 'white' THEN game.players.white.user.id 
              WHEN game.winner = 'black' THEN game.players.black.user.id
          END, 
          'no_winner'  // Fallback value if winner is null or 'draw'
      )

  // 3. Establish Game Relationships
  MERGE (g)-[:OPENING]->(o)
  MERGE (g)-[:WHITE_PLAYER]->(w)
  MERGE (g)-[:BLACK_PLAYER]->(b)

  // 4. Batch Process FENs and Moves for this specific game
  WITH g, game
  UNWIND range(0, size(game.moves_final) - 1) AS i
  WITH i, game.moves_final[i] AS current_move, g, game

  // Reuse global FEN nodes
  MERGE (f:FEN {fen: current_move.fen})

  // Create GameMove nodes unique to this game AND this move number
  // IMPORTANT: Added moveNumber to the MERGE key to ensure moves don't overwrite each other
  MERGE (gm:GameMove {gameId: g.id, moveNumber: i + 1})
  SET gm.san = current_move.san,
      gm.movingSide = CASE WHEN i % 2 = 0 THEN 'white' ELSE 'black' END

  MERGE (gm)-[:POSITION_REACHED]->(f) 

  // 5. Handle Start Position
  FOREACH (_ IN CASE WHEN i = 0 THEN [1] ELSE [] END |
      MERGE (g)-[:FIRST_MOVE]->(gm)
  )

  // 6. Chain the moves together
  WITH i, gm, game, g
  WHERE i < size(game.moves_final) - 1
  WITH i, gm, g, game.moves_final[i+1] AS next_move_data, i + 2 AS next_move_num
  MERGE (next_gm:GameMove {gameId: g.id, moveNumber: next_move_num})
  SET next_gm.san = next_move_data.san
  MERGE (gm)-[:NEXT_MOVE]->(next_gm)
"""

FETCH_USER_GAMES_QUERY = """
  MATCH (p:Player {name: $playerId}) 
  MATCH (g: Game)-[r1:WHITE_PLAYER|BLACK_PLAYER]->(p)
  MATCH (g)-[r2:OPENING]-(o:Opening)
  return p,g,o,r1,r2
  LIMIT 200
"""


@st.cache_resource
def get_lichess_client():
	li_session = berserk.TokenSession(st.secrets.LICHESS_API_TOKEN)
	return berserk.Client(session=li_session)

li_client = get_lichess_client()
board = chess.Board()

# @st.cache_data(ttl=86400)
def fetch_games_for_user(username):
  games_iterator = li_client.games.export_by_player(
    username=username,
    perf_type="blitz,rapid",
    max=st.secrets.GAME_FETCH_COUNT,
    opening=True,
    literate=True,
    tags=True
  )
  return [x for x in games_iterator] 

def process_game(game):
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

def load_games(li_username):
  t1 = time.time()
  st.write("Fetching games of the last week from Lichess")
  
  print(f"Fetching lichess games for {li_username}")
  games = fetch_games_for_user(li_username)
  print(f"Fetched {len(games)} for user {li_username}")
  
  st.write(f"Fetched {len(games)} games")
  st.write("Processing Games")
  
  print("Processing games")
  games_processed = [process_game(game) for game in games]
  print("Games processed")
  
  st.write("Games processed")
  st.write("Create knowledge Graph")
  
  print("Inserting games in neo4j")
  insert_result = execute_db_query(GAMES_INSERT_QUERY, params=dict(games=games_processed))
  print(f"Games Inserted. Insertion Summary: {insert_result.summary.counters}")
  
  # This code snippet is fetching game data from the database for visualization purposes.
  # This code snippet is fetching game data from the database for visualization purposes.
  print("Fetching game data for visualization")
  game_data_graph = execute_db_query(FETCH_USER_GAMES_QUERY, params=dict(playerId=li_username), result_transformer=Result.graph)
  print(f"Data Fetched. Received transformed graph object")
  
  print("Building Visualization graph")
  vg = neo4j.from_neo4j(game_data_graph)
  print(f"Visualization graph built with {len(vg.nodes)} nodes and {len(vg.relationships)} relations")
  
  st.write("Knowledge graph ready")
  t2 = time.time()
  print(f"Loaded games in {t2-t1} seconds")
  return vg