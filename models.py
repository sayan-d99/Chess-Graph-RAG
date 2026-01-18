from enum import StrEnum
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd


class Node1ClassificationOutput(BaseModel):
  """Classification logic for the first gate"""
  category: Literal["invalid", "simple", "complex"]
  response: str
  
class ChartData(BaseModel):
  """Structure for chart rendering"""
  chart_type: Literal["area", "line", "scatter", "bar"]
  labels: List[str]
  values: List[float]
  x_label: str
  y_label: str
  
class FinalResponse(BaseModel):
  """Final response of the chain to be used by the UI"""
  response: str
  has_chart: bool
  chart_data: Optional[ChartData]
  suggestions: List[str]
  
class ChatMessage:
  def __init__(self, role, text, has_chart, chart_type, chart_df):
    self.role = role
    self.text = text
    self.has_chart = has_chart
    self.chart_type = chart_type
    self.chart_df = chart_df
    
  def render(self, st):
    with st.chat_message(self.role):
      st.markdown(self.text)
      if self.has_chart:
        if self.chart_type == "area":
          st.area_chart(self.chart_df)
        elif self.chart_type == "bar":
          st.bar_chart(self.chart_df)
        elif self.chart_type == "scatter":
          st.scatter_chart(self.chart_df)
        elif self.chart_type == "line":
          st.line_chart(self.chart_df)
      
  @classmethod    
  def from_llm_response(cls, response):
    response_text = response.response
    has_chart = response.has_chart
    chart_type = ""
    chart_df = None
    if has_chart:
      chart_data = response.chart_data
      chart_type = chart_data.chart_type
      chart_df = pd.DataFrame({
        chart_data.x_label: chart_data.labels,
        chart_data.y_label: chart_data.values
      })
    return cls(role="ai", text=response_text, has_chart=has_chart, chart_type=chart_type, chart_df=chart_df)
  
class MoveAnalysisJudgement(StrEnum):
  INACCURACY = "Inaccuracy"
  MISTAKE = "Mistake"
  BLUNDER = "Blunder"
  MATE = "Mate"  
  NONE = 'NotAvailable'

class Side(StrEnum):
  WHITE = "white"
  BLACK = "black"

class ChessOpening:
  def __init__(self, name:str, eco:str):
    self.name = name
    self.eco = eco
  
  def __repr__(self):
    return f"GameOpening[name:{self.name},eco:{self.eco}]"

class ChessMove:
  def __init__(self, move_san, move_fen, analysis_type, move_number, moving_side, move_eval, forced_mate):
    self.move_san = move_san
    self.move_fen = move_fen
    self.eval = move_eval
    self.forced_mate = forced_mate
    self.analysis_type = analysis_type
    self.move_number= move_number
    self.moving_side= moving_side
  
  def __repr__(self):
    return f"ChessMove[move_san={self.move_san},move_fen={self.move_fen},analysis_type={self.analysis_type},move_number={self.move_number},forced_mate={self.forced_mate},moving_side={self.moving_side},eval={self.eval}]"
  
class ChessGame:
  def __init__(self, 
               game_id: str, 
               white_id: str, 
               black_id: str, 
               winner_id: str, 
               winning_side: Side,
               played_on, 
               status: str,
               game_speed: str,
               opening: ChessOpening, 
               n_moves: int = 0,
               moves: List[ChessMove] = []): 
    self.game_id: str = game_id
    self.white_id: str = white_id
    self.game_speed: str = game_speed
    self.black_id: str = black_id
    self.n_moves: int = n_moves
    self.opening: ChessOpening = opening
    self.played_on=played_on
    self.moves: List[ChessMove] = moves
    self.winner_id: str = winner_id
    self.winning_side: Side = winning_side
    self.status: str = status
  
  def to_dict(self):
    return {
        "game_id": self.game_id,
        "white_id": self.white_id,
        "black_id": self.black_id,
        "winner_id": self.winner_id,
        "winning_side": str(self.winning_side),
        "played_on": self.played_on,
        "status": self.status,
        "game_speed": self.game_speed,
        "perf": self.game_speed, 
        "opening": {
            "name": self.opening.name,
            "eco": self.opening.eco
        },
        "moves": [
            {
                "move_san": m.move_san,
                "move_fen": m.move_fen,
                "eval": m.eval,
                "is_forced_mate": m.forced_mate,
                "analysis_type": str(m.analysis_type),
                "move_number": m.move_number,
                "moving_side": str(m.moving_side),
            } for m in self.moves
        ]
    }
    
  def __repr__(self):
    return f"ChessGame[game_id:{self.game_id},white_id:{self.white_id},black_id={self.black_id},played_on:{self.played_on},winner_id:{self.winner_id},winning_side:{self.winning_side},game_speed:{self.game_speed},n_moves:{self.n_moves}opening={self.opening},moves:{str(self.moves)}]"