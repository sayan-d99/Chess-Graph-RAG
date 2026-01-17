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