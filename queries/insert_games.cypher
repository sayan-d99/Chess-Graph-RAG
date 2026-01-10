UNWIND $games AS game
MERGE (w:Player {pid: game.players.white.user.id})
ON CREATE SET w.name = game.players.white.user.name

MERGE (b:Player {pid: game.players.black.user.id})
ON CREATE SET b.name = game.players.black.user.name 

MERGE (g:Game {id: game.id})
SET g.gameType = game.perf,
    g.status = game.status,
    g.winningSide = coalesce(game.winner, "NO_WINNER"),
    g.fullId = game.fullId,
    g.playedOn = game.createdAt,
    g.winnerId = coalesce(
        CASE 
            WHEN game.winner = 'black' THEN game.players.black.user.id 
            WHEN game.winner = 'white' THEN game.players.white.user.id
        END, 
        'no_winner'  // Fallback value if winner is null or 'draw'
    )

MERGE (g)-[:WHITE_PLAYER]->(w)
MERGE (g)-[:BLACK_PLAYER]->(b)

FOREACH (_ IN CASE WHEN game.opening IS NOT NULL THEN [1] ELSE [] END | 
  MERGE (o: Opening {eco: game.opening.eco})
  ON CREATE SET o.name = game.opening.name
  MERGE (g)-[:OPENING]->(o)
)

WITH g, game
UNWIND range(0, size(coalesce(game.moves_final, [])) - 1) AS i
WITH i, game.moves_final[i] AS current_move, g, game

MERGE (f:FEN {fen: current_move.fen})
MERGE (gm:GameMove {gameId: g.id, moveNumber: i + 1})
SET gm.san = current_move.san,
    gm.movingSide = CASE WHEN i % 2 = 0 THEN 'white' ELSE 'black' END

MERGE (gm)-[:POSITION_REACHED]->(f) 

FOREACH (_ IN CASE WHEN i = 0 THEN [1] ELSE [] END |
  MERGE (g)-[:FIRST_MOVE]->(gm)
)

WITH i, gm, game, g
WHERE i < size(game.moves_final) - 1
WITH i, gm, g, game.moves_final[i+1] AS next_move_data, i + 2 AS next_move_num
MERGE (next_gm: GameMove {gameId: g.id, moveNumber: next_move_num})
SET next_gm.san = next_move_data.san
MERGE (gm)-[:NEXT_MOVE]->(next_gm)


// WITH g, game
// UNWIND range(0, size(game.moves_final) - 1) AS i
// WITH i, game.moves_final[i] AS move_data, g

// // 1. Create the Move Node and the FEN Node
// MERGE (f:FEN {fen: move_data.fen})
// MERGE (gm:GameMove {gameId: g.id, moveNumber: i + 1})
// SET gm.san = move_data.san,
//     gm.movingSide = CASE WHEN i % 2 = 0 THEN 'white' ELSE 'black' END

// MERGE (gm)-[:POSITION_REACHED]->(f) 

// // 2. Link Game to First Move
// FOREACH (_ IN CASE WHEN i = 0 THEN [1] ELSE [] END |
//     MERGE (g)-[:FIRST_MOVE]->(gm)
// )

// // 3. Link to PREVIOUS move (More efficient than looking forward)
// WITH g, i, gm
// WHERE i > 0
// MATCH (prev_gm:GameMove {gameId: g.id, moveNumber: i})
// MERGE (prev_gm)-[:NEXT_MOVE]->(gm)