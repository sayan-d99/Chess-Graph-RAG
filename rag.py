import sys
import os
import torch
import chess
import numpy as np
import streamlit as st
from langchain.embeddings.base import Embeddings
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_outputs import Node1ClassificationOutput, FinalResponse
from typing import List, Any


# EMBEDDING CLASS SETUP
ENCODER_PATH = os.path.join(os.getcwd(), "Encoder-ChessLM")
if ENCODER_PATH not in sys.path:
    sys.path.append(ENCODER_PATH)

def load_prompt(filename):
  with open(filename, 'r') as f:
    return f.read()

CYPHER_QUERY_PROMPT = load_prompt("prompts/cypher_generation.txt")
DECISION_PROMPT = load_prompt("prompts/decision_prompt.txt")
RESPONSE_GENERATOR_PROMPT = load_prompt("prompts/response_generator.txt")

try:
  # Importing the exact model class from the user's train.py
  from train.train import ChessVisionTransformer
except ImportError:
  # Fallback if running directly inside the repo root
  try:
    from train.train import ChessVisionTransformer
  except ImportError as e:
    raise ImportError(
      f"Could not find 'train/train.py'. Ensure you are in the correct directory. Error: {e}"
    )

class ChessLMEmbeddings(Embeddings):
  def __init__(self, model_path: str = None, device: str = None):
      """
      Args:
          model_path: Path to the .safetensors or .pt file.
          device: 'cuda' or 'cpu'.
      """
      self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
      # 1. Initialize Model with EXACT config from train.py main() function
      self.model = ChessVisionTransformer(
          d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1
      )
      # 2. Load Weights
      if model_path and os.path.exists(model_path):
          print(f"Loading weights from {model_path}...")
          # Handle Safetensors or Standard PyTorch
          if model_path.endswith(".safetensors"):
              from safetensors.torch import load_file

              state_dict = load_file(model_path)
          else:
              checkpoint = torch.load(model_path, map_location=self.device)
              # If the file contains optimizer states (like in train.py), extract model state
              if "model_state_dict" in checkpoint:
                  state_dict = checkpoint["model_state_dict"]
              else:
                  state_dict = checkpoint
          # Remove keys that might cause mismatches (like 'module.' from DDP)
          state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
          # Load
          msg = self.model.load_state_dict(state_dict, strict=False)
          print(f"Weights loaded. {msg}")
      else:
          print(
              "WARNING: No model_path found. Using random weights (Embeddings will be garbage)."
          )
      self.model.to(self.device)
      self.model.eval()

  def _preprocess(self, fen: str):
      """
      Replicates the logic from data/preprocessing.py and train/train.py
      """
      # print(f"ChessLMEmbeddings[_preprocess]: FEN - {fen}")
      # --- FIX: Handle LangChain's dimension check ---
      if fen == "foo":
          # LangChain sends "foo" to check vector dimension.
          # We swap it for the starting position to prevent a crash.
          fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      if "fen:" in fen:
          fen = fen[fen.index("fen:") + 4 :]
          # print(
          #     f"ChessLMEmbeddings[_preprocess]: FEN (after removing prefix) - {fen}"
          # )
      board = chess.Board(fen)
      # --- A. Piece Mapping (from preprocessing.py) ---
      piece_values = {
          "P": 1,
          "N": 2,
          "B": 3,
          "R": 4,
          "Q": 5,
          "K": 6,  # White
          "p": -1,
          "n": -2,
          "b": -3,
          "r": -4,
          "q": -5,
          "k": -6,  # Black
      }
      # --- B. Create Matrix (from preprocessing.py) ---
      # Note: train.py expects float32
      matrix = np.zeros((8, 8), dtype=np.float32)
      for square in chess.SQUARES:
          piece = board.piece_at(square)
          if piece is not None:
              rank = chess.square_rank(square)
              file = chess.square_file(square)
              matrix[rank, file] = piece_values[piece.symbol()]
      # --- C. Flatten (for train.py: encode_board) ---
      # train.py line: x = board_state.view(batch_size, 64, 1)
      # We flatten to 64 here.
      flat_board = matrix.flatten()
      # --- D. Turn Handling (from preprocessing.py) ---
      # preprocessing.py: turn = int(not board.board.turn)
      # White(True) -> 0, Black(False) -> 1
      turn_val = int(not board.turn)
      return (
          torch.tensor(flat_board, dtype=torch.float32)
          .unsqueeze(0)
          .to(self.device),  # [1, 64]
          torch.tensor([turn_val], dtype=torch.long).to(self.device),  # [1]
      )

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    results = []
    with torch.no_grad():
      for fen in texts:
        # 1. Preprocess
        board_tensor, turn_tensor = self._preprocess(fen)
        # 2. Forward Pass
        # train.py uses 'encode_board' which requires a task token.
        # We use 'mpp_token' (Masked Piece Prediction) as it represents
        # understanding the static board state.
        x = self.model.encode_board(
            board_tensor, turn_tensor, self.model.mpp_token
        )
        # Run through transformer
        x = self.model.transformer(x)
        # 3. Extract Embedding
        # In train.py: task_output = model.layer_norm(x[:, 0])
        # The token at index 0 is the task token (our CLS token equivalent)
        embedding = self.model.layer_norm(x[:, 0])
        results.append(embedding.squeeze().cpu().tolist())
    return results

  def embed_query(self, text: str) -> List[float]:
    return self.embed_documents([text])[0]

# EMBEDDING CLASS OBJECT
embedding_model = ChessLMEmbeddings(model_path="./model.safetensors")

def generate_fen_embeddings():
  return Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=st.secrets.NEO4J_URI,
    username=st.secrets.NEO4J_USERNAME,
    password=st.secrets.NEO4J_PASSWORD,
    index_name="fen_embeddings",
    node_label="FEN",
    text_node_properties=["fen"],
    embedding_node_property="embedding"
  )

@st.cache_resource
def setup_graph_and_vector():
  graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"]
  )
  vector_store = generate_fen_embeddings()
  return vector_store, embedding_model, graph

@st.cache_resource
def get_model_obj():
  chat_model = ChatGoogleGenerativeAI(
    model=st.secrets.GEMINI_MODEL,
    api_key=st.secrets.GEMINI_API_KEY)
  return chat_model

# cypher_prompt = PromptTemplate(
# 	input_variables=["schema", "question", "corrections", "username"],
# 	template=CYPHER_QUERY_PROMPT
# )

chat_model = get_model_obj()
vector_store, embedding_model, graph = setup_graph_and_vector()

node1_prompt =  ChatPromptTemplate.from_messages([
    ("system", DECISION_PROMPT),
    ("human", "{query}")
])
node1_chain = node1_prompt | chat_model.with_structured_output(Node1ClassificationOutput)

node2_prompt = ChatPromptTemplate([
  ("system", CYPHER_QUERY_PROMPT),
  ("human", "Answer the following question for the player with pid : {username} - {query}")
])
node2 = GraphCypherQAChain.from_llm(
	llm=chat_model,
  vector_store=vector_store,
  graph=graph,
  verbose=True,
  cypher_prompt=node2_prompt,
  return_intermediate_steps=True,
  return_direct=True,
  allow_dangerous_requests=True
)

node3_prompt = ChatPromptTemplate.from_messages([
  ("system", RESPONSE_GENERATOR_PROMPT),
  ("human", "Query: {query}\nDatabase Data: {raw_data}")
])
node3_chain = node3_prompt | chat_model.with_structured_output(FinalResponse)

def execute_rag_query(user_prompt, username):
  print(f"Executing Prompt: {user_prompt} for user : {username}")
  node1_output = node1_chain.invoke({"query": user_prompt})
  print(f"Node 1 Output: {node1_output}")
  if node1_output.category != "complex":
    return FinalResponse(
      category=node1_output.category,
      text_res=node1_output.response,
      has_chart=False,
      chart_data=None,
      suggestions=[] 
    )
  
  node2_output = node2.invoke({"query": user_prompt, "username": username})
  print(f"Node 2 Output: {node2_output}")
  raw_data = node2_output['result']
  
  if not raw_data:
    return FinalResponse(
      category="no_data", 
      text_res="Could not find any data relevant to your query. Please ask another question",
      has_chart=False,
      chart_data=None,
      suggestions=[]
    )
  
  node3_output = node3_chain.invoke({"query": user_prompt, "raw_data": raw_data})
  print(f"Node 3 Output: {node3_output}")
  return node3_output