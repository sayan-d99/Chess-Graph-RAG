from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

class Node1ClassificationOutput(BaseModel):
  """Classification logic for the first gate"""
  category: Literal["invalid", "simple", "complex"]
  response: str
  
class ChartData(BaseModel):
  """Structure for chart rendering"""
  chart_type: Literal["area", "line", "scatter", "bar"]
  labels: List[str]
  values: List[float]
  
class FinalResponse(BaseModel):
  """Final response of the chain to be used by the UI"""
  category: Literal["invalid", "simple", "complex", "no_data"]
  text_res: str
  has_chart: bool
  chart_data: Optional[ChartData]
  suggestions: List[str]
  
  