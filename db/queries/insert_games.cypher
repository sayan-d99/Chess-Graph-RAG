UNWIND $games AS game
MERGE (w:Player {pid: game.white_id})

MERGE (b:Player {pid: game.black_id})

MERGE (g:Game {id: game.game_id})
ON CREATE SET g.status = game.status,
    g.winningSide = game.winning_side,
    g.playedOn = game.played_on,
    g.winnerId = game.winner_id,
    g.gameSpeed = game.game_speed

MERGE (g)-[:WHITE_PLAYER]->(w)
MERGE (g)-[:BLACK_PLAYER]->(b)
 
MERGE (o: Opening {eco: game.opening.eco})
ON CREATE SET o.name = game.opening.name

MERGE (g)-[:OPENING]->(o)

WITH g, game
UNWIND range(0, size(game.moves) - 1) AS i
WITH i, game.moves[i] AS current_move, g, game
MERGE (f:FEN {fen: current_move.move_fen})
MERGE (gm:GameMove {gameId: game.game_id, moveNumber: current_move.move_number})
ON CREATE SET gm.san = current_move.move_san,
              gm.movingSide = current_move.moving_side,
              gm.judgementRemark = current_move.analysis_type,
              gm.isForcedMate = current_move.is_forced_mate,
              gm.eval = current_move.eval
MERGE (gm)-[:POSITION_REACHED]->(f) 

FOREACH (_ IN CASE WHEN i = 0 THEN [1] ELSE [] END |
  MERGE (g)-[:FIRST_MOVE]->(gm)
)

WITH i, gm, game, g
WHERE i < size(game.moves) - 1
WITH i, gm, g, game.moves[i+1] AS next_move, i + 2 AS next_move_num
MERGE (next_gm: GameMove {gameId: g.id, moveNumber: next_move_num})
MERGE (gm)-[:NEXT_MOVE]->(next_gm)