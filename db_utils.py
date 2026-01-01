from neo4j import GraphDatabase
import streamlit as st
from chess_utils import get_game_players

@st.cache_resource()
def get_db_driver():
  return GraphDatabase.driver(
    uri=st.secrets.NEO4J_URI, 
    auth=(st.secrets.NEO4J_USERNAME, st.secrets.NEO4J_PASSWORD)
  )

db_driver = get_db_driver()

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

def insert_games(list_games):
  insert_result = None
  try:
    insert_result = db_driver.execute_query(GAMES_INSERT_QUERY,games=list_games)
  except Exception as e:
    print(f"Exception occurred when inserting games {e}")
  return insert_result