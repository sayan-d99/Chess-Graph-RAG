# import sys
# import os
# import torch
# import chess
# import numpy as np
# import streamlit as st
# from langchain.embeddings.base import Embeddings
# from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from typing import List, Any

# ENCODER_PATH = os.path.join(os.getcwd(), "Encoder-ChessLM")
# if ENCODER_PATH not in sys.path:
#     sys.path.append(ENCODER_PATH)

# try:
#     # Importing the exact model class from the user's train.py
#     from train.train import ChessVisionTransformer
# except ImportError:
#     # Fallback if running directly inside the repo root
#     try:
#         from train.train import ChessVisionTransformer
#     except ImportError as e:
#         raise ImportError(
#           f"Could not find 'train/train.py'. Ensure you are in the correct directory. Error: {e}"
#         )

# class ChessLMEmbeddings(Embeddings):
#     def __init__(self, model_path: str = None, device: str = None):
#         """
#         Args:
#             model_path: Path to the .safetensors or .pt file.
#             device: 'cuda' or 'cpu'.
#         """
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # 1. Initialize Model with EXACT config from train.py main() function
#         self.model = ChessVisionTransformer(
#             d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1
#         )
#         # 2. Load Weights
#         if model_path and os.path.exists(model_path):
#             print(f"Loading weights from {model_path}...")
#             # Handle Safetensors or Standard PyTorch
#             if model_path.endswith(".safetensors"):
#                 from safetensors.torch import load_file

#                 state_dict = load_file(model_path)
#             else:
#                 checkpoint = torch.load(model_path, map_location=self.device)
#                 # If the file contains optimizer states (like in train.py), extract model state
#                 if "model_state_dict" in checkpoint:
#                     state_dict = checkpoint["model_state_dict"]
#                 else:
#                     state_dict = checkpoint
#             # Remove keys that might cause mismatches (like 'module.' from DDP)
#             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#             # Load
#             msg = self.model.load_state_dict(state_dict, strict=False)
#             print(f"Weights loaded. {msg}")
#         else:
#             print(
#                 "WARNING: No model_path found. Using random weights (Embeddings will be garbage)."
#             )
#         self.model.to(self.device)
#         self.model.eval()

#     def _preprocess(self, fen: str):
#         """
#         Replicates the logic from data/preprocessing.py and train/train.py
#         """
#         print(f"ChessLMEmbeddings[_preprocess]: FEN - {fen}")
#         # --- FIX: Handle LangChain's dimension check ---
#         if fen == "foo":
#             # LangChain sends "foo" to check vector dimension.
#             # We swap it for the starting position to prevent a crash.
#             fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#         if "fen:" in fen:
#             fen = fen[fen.index("fen:") + 4 :]
#             # print(
#             #     f"ChessLMEmbeddings[_preprocess]: FEN (after removing prefix) - {fen}"
#             # )
#         board = chess.Board(fen)
#         # --- A. Piece Mapping (from preprocessing.py) ---
#         piece_values = {
#             "P": 1,
#             "N": 2,
#             "B": 3,
#             "R": 4,
#             "Q": 5,
#             "K": 6,  # White
#             "p": -1,
#             "n": -2,
#             "b": -3,
#             "r": -4,
#             "q": -5,
#             "k": -6,  # Black
#         }
#         # --- B. Create Matrix (from preprocessing.py) ---
#         # Note: train.py expects float32
#         matrix = np.zeros((8, 8), dtype=np.float32)
#         for square in chess.SQUARES:
#             piece = board.piece_at(square)
#             if piece is not None:
#                 rank = chess.square_rank(square)
#                 file = chess.square_file(square)
#                 matrix[rank, file] = piece_values[piece.symbol()]
#         # --- C. Flatten (for train.py: encode_board) ---
#         # train.py line: x = board_state.view(batch_size, 64, 1)
#         # We flatten to 64 here.
#         flat_board = matrix.flatten()
#         # --- D. Turn Handling (from preprocessing.py) ---
#         # preprocessing.py: turn = int(not board.board.turn)
#         # White(True) -> 0, Black(False) -> 1
#         turn_val = int(not board.turn)
#         return (
#             torch.tensor(flat_board, dtype=torch.float32)
#             .unsqueeze(0)
#             .to(self.device),  # [1, 64]
#             torch.tensor([turn_val], dtype=torch.long).to(self.device),  # [1]
#         )

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         results = []
#         with torch.no_grad():
#             for fen in texts:
#                 # 1. Preprocess
#                 board_tensor, turn_tensor = self._preprocess(fen)
#                 # 2. Forward Pass
#                 # train.py uses 'encode_board' which requires a task token.
#                 # We use 'mpp_token' (Masked Piece Prediction) as it represents
#                 # understanding the static board state.
#                 x = self.model.encode_board(
#                     board_tensor, turn_tensor, self.model.mpp_token
#                 )
#                 # Run through transformer
#                 x = self.model.transformer(x)
#                 # 3. Extract Embedding
#                 # In train.py: task_output = model.layer_norm(x[:, 0])
#                 # The token at index 0 is the task token (our CLS token equivalent)
#                 embedding = self.model.layer_norm(x[:, 0])
#                 results.append(embedding.squeeze().cpu().tolist())
#         return results

#     def embed_query(self, text: str) -> List[float]:
#         return self.embed_documents([text])[0]

# @st.cache_resource
# def setup_graph_and_vector():
#   embedding_model = ChessLMEmbeddings(model_path="./model.safetensors")
#   graph = Neo4jGraph(
#     url=st.secrets["NEO4J_URI"],
#     username=st.secrets["NEO4J_USERNAME"],
#     password=st.secrets["NEO4J_PASSWORD"]
#   )
#   vector_store = Neo4jVector.from_existing_graph(
#     embedding=embedding_model,
#     url=st.secrets.NEO4J_URI,
#     username=st.secrets.NEO4J_USERNAME,
#     password=st.secrets.NEO4J_PASSWORD,
#     index_name="fen_embeddings",
#     node_label="FEN",
#     text_node_properties=["fen"],
#     embedding_node_property="embedding"
#   )
#   return vector_store, graph

# @st.cache_resource
# def get_model_obj():
#   llm = HuggingFaceEndpoint(
#     repo_id=st.secrets.HUGGINGFACE_MODEL,
#     verbose=False,
#     max_new_tokens=2000,
#     temperature=0,
#     repetition_penalty=1.1,
#     huggingfacehub_api_token=st.secrets.HUGGINGFACEHUB_API_TOKEN
#   )
#   chat_model = ChatHuggingFace(llm=llm)
#   return llm, chat_model

# CORRECTIONS = [
#   """
# 		When fetching multiple relations at once, do not use `:` before every relation. This feature is deprecated in Neo4j. Use `:` only before the first relation. For example,
# 		MATCH (g: Game)-[:WHITE_PLAYER|BLACK_PLAYER]->(p: Player)
#   """,
#  	"""When querying for games played by a player, the query generated currently is
# 		MATCH (p:Player)-[:WHITE_PLAYER|:BLACK_PLAYER]->(g:Game)
# 		This is wrong. The relation WHITE_PLAYER and BLACK_PLAYER are from the Game node to the Player node.
# 		Hence, the correct query is 
# 		MATCH (g:Game)-[:WHITE_PLAYER|BLACK_PLAYER]->(p: Player)
#  	"""
# ]

# CYPHER_GENERATION_TEMPLATE = """
# You are an expert Neo4j Developer and Chess Analyst.
# Your task is to answer the user's question regarding chess analytics. Convert the user's natural language question into the most relevant and valid Cypher query as required and use it to gather data to answer the user's query.
# You are given a Graph DB which contains the chess game data you are supposed to analyze based on the user query.

# CRITICAL: Output ONLY the Cypher query. 
# Start your response directly with the word MATCH or CALL. 
# Do not use <think> tags. Do not explain your code.

# THE GRAPH DB SCHEMA (Strictly enforce this)
# Do not guess the schema. Use ONLY the nodes and relationships defined below:

# Auto-generated schema:
# {schema}

# ### 1. Detailed Description of the Nodes & Properties in the schema (STRICTLY FOLLOW THE BELOW SCHEMA DESCRIBED. DO NOT USE ANY NODES AND ATTRIBUTES OUTSIDE OF THE ONES MENTIONED BELOW)
# - Game
#   - `id` (String). Unique Identifier of the Game
#   - `fullId` (String) 
#   - `status` (String). It can be one of the following values - 'created','started','aborted','mate' (value when game ends in checkmate),'resign','stalemate','timeout','draw','outoftime','cheat','noStart','unknownFinish','insufficientMaterialClaim'
#   - `result` (String): e.g., "1-0" (White wins), "0-1" (Black wins), "1/2-1/2" (game is a draw).
#   - `winingSide` (String) - The side which won the game. Values can be 'white' or 'black'. If there is no winner due to any reason such as draw, the attribute will have the value '__no__winner__'. Check the `winningId` attribute for more information.
#   - `winnerId` (String) - The id of the player who won the game. This value will be same as pid attribute on Player Node
#   - `date` (Date): Date of the game.
# - GameMove
#   - `san` (String): Standard Algebraic Notation (e.g., "Nf3", "e4").
#   - `moveNumber` (Integer): The move number.
#   - `movingSide` (String): The moving side. Values can be 'white' or 'black'
# - FEN
#   - `fen` (String): The FEN string of the board position.
#   - `eval` (FLoat): The engine evaluation of the FEN position. If `eval` is not present for a particular FEN Node, this field will contain null value. Ignore such nodes while performing calculation on `eval` attribute. Mention that such nodes have been ignored in the output
#   - `embedding` (List<Float>): The vector representation (do not query this directly).
# - Player
#   - `pid` (String): ID of the player.
#   - `name` (String): Name of the player.
# - Opening
#   - `name` (String): Name of the opening (e.g., "Sicilian Defense").
#   - `eco` (String): The code (e.g., "B90").

# Relationships:
# - (:Game)-[:WHITE_PLAYER]->(:Player)
# - (:Game)-[:BLACK_PLAYER]->(:Player)
# - (:Game)-[:OPENING]->(:Opening)
# - (:Game)-[:FIRST_MOVE]->(:GameMove)
# - (:GameMove)-[:NEXT_MOVE]->(:GameMove)
# - (:GameMove)-[:POSITION_REACHED]->(:FEN)

# ### 2. HOW TO USE RELATIONS?
# Here are some examples about how to use the above defined relations to fetch different kinds of data 
# - Fetch Games of a Player.
# 	Cypher: MATCH (g:Game) -[:WHITE_PLAYER|BLACK_PLAYER] -> (p: Player)
# - Fetch Games of a Player played as white
# 	Cypher: MATCH (g:Game) -[:WHITE_PLAYER] -> (p: Player)
# - Fetch Games of a Player played as black
# 	Cypher: MATCH (g:Game) -[:BLACK_PLAYER] -> (p: Player)
# - Fetch the type of opening played in a game
# 	Cypher: MATCH (g:Game) -[:OPENING] -> (p: Opening)
# - Fetch the type first move played in the game
# 	Cypher: MATCH (g:Game) -[:FIRST_MOVE] -> (gm: GameMove)
# - Fetch the games played in a Game
# 	Cypher: MATCH (g:Game) -[:FIRST_MOVE]->(gm: GameMove)
# 	MATCH (gm: GameMove) -[:NEXT_MOVE]->(gm: GameMove) // Traverse this relation recursively until you run out of GameMove nodes


# ### 3. LOGIC & RULES
# - CRITICAL: Output ONLY the Cypher query. Start your response directly with the word MATCH or CALL. Do not use <think> tags. Do not explain your code. 
# -Similarity Search: If the user asks for "similar positions", "positions like this", or "positional themes", you MUST use the vector index.
#   - Syntax: `CALL db.index.vector.queryNodes('fen_embeddings', 10, $embedding) YIELD node AS fen, score`
# - Game Moves: For every move of a game, a GameMove Node is created. The Game node is related to the first GameMove using the 'START_POSITION' relation. GameMove nodes are related to each other using 'NEXT_MOVE' relation.
# - Traversing Moves: To find what happened after a position, traverse: `(fen)<-[:POSITION_REACHED]-(gm)-[:NEXT_MOVE]->(next_gm)`
# - Evaluation: `eval` is from White's perspective. High positive = White winning. High negative = Black winning.

# ### 4. MANUAL PROGRAMMER FEEDBACK AND CORRECTIONS
# The following rules were established to correct previous model errors. 
# Follow these strictly:
# {corrections}

# -------------------------------------------------------

# {question}
# """

# cypher_prompt = PromptTemplate(
# 	input_variables=["schema", "question", "corrections"],
# 	template=CYPHER_GENERATION_TEMPLATE
# )

# llm, chat_model = get_model_obj()
# vector_store, graph = setup_graph_and_vector()

# chain = GraphCypherQAChain.from_llm(
# 	llm=chat_model,
#   vector_store=vector_store,
#   graph=graph,
#   verbose=True,
#   cypher_prompt=cypher_prompt,
#   return_intermediate_steps=True,
#   allow_dangerous_requests=True
# )

# def execute_rag_query(query, username):
#   final_query = f"For the Player with id: '{username}', answer the question: {query}"
#   print(f"Firing query {final_query}")
#   response = chain.invoke({"query": final_query, "corrections": CORRECTIONS})
#   return response
