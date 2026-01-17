import torch
import chess
from typing import List, Any
import sys
import os
import numpy as np
from langchain.embeddings.base import Embeddings

# EMBEDDING CLASS SETUP
ENCODER_PATH = os.path.join(os.getcwd(), "Encoder-ChessLM")
if ENCODER_PATH not in sys.path:
    sys.path.append(ENCODER_PATH)

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
