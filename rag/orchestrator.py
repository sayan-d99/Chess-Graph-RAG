import streamlit as st

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, trim_messages
from models import Node1ClassificationOutput, FinalResponse
from rag.embedding import ChessLMEmbeddings
from util import load_file
from models import ChatMessage
from typing import List
import tiktoken

print("Loading prompts")
CYPHER_QUERY_PROMPT = load_file("rag/prompts/cypher_generation.txt")
DECISION_PROMPT = load_file("rag/prompts/decision_prompt.txt")
RESPONSE_GENERATOR_PROMPT = load_file("rag/prompts/response_generator.txt")
print("Prompts loaded")

# SETUP FUNCTIONS

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
  return vector_store, graph

@st.cache_resource
def get_model_obj():
  reasoning = {
    "effort": st.secrets.OPENAI_RSEFF,
    "summary": None
  }
  chat_model = ChatOpenAI(
    model=st.secrets.OPENAI_MODEL,
    api_key=st.secrets.OPENAI_KEY,
    reasoning=reasoning,
    # temperature=0
  )
  return chat_model

def get_token_count(messages) -> int:
  text = " ".join([m.content for m in messages])
  return len(token_counter_encoder.encode(text))

# GLOBAL VARIABLES
chat_model = get_model_obj()
vector_store, graph = setup_graph_and_vector()
token_counter_encoder = tiktoken.get_encoding("o200k_base")
message_history_trimmer = trim_messages(
  max_tokens=4000,
  strategy="last",
  token_counter=get_token_count,
  include_system=True,
  start_on="human",
  allow_partial=False
)

# CREATE THE NODES OF THE CHAINS
def get_message_history(messages: List[ChatMessage]) -> str:
  lc_messages = [HumanMessage(content=m.text) if m.role == "human" else AIMessage(content=m.text) for m in messages]
  lc_messages_trimmed = message_history_trimmer.invoke(lc_messages)
  history = ""
  for m in lc_messages_trimmed:
    role = "User" if isinstance(m, HumanMessage) else "Assistant"
    history += f"Role: {role} Message: {m.content}\n"
  return history

node1_prompt =  ChatPromptTemplate.from_messages([
    ("system", DECISION_PROMPT + "\n\n Here is the past conversation history: {history}"),
    ("human", "{query}")
])
node1_chain = node1_prompt | chat_model.with_structured_output(Node1ClassificationOutput)

node2_prompt = ChatPromptTemplate([
  ("system", CYPHER_QUERY_PROMPT + "\n\n Here is the past conversation history: {history}"),
  ("human", "Answer the following question for the player with pid : {pid} - {query}")
])
node2 = GraphCypherQAChain.from_llm(
	llm=chat_model,
  vector_store=vector_store,
  graph=graph,
  verbose=True,
  cypher_prompt=node2_prompt,
  return_intermediate_steps=True,
  return_direct=True,
  allow_dangerous_requests=True,
  top_k=200
)

node3_prompt = ChatPromptTemplate.from_messages([
  ("system", RESPONSE_GENERATOR_PROMPT),
  ("human", "Query: {query}\nDatabase Data: {raw_data}")
])
node3_chain = node3_prompt | chat_model.with_structured_output(FinalResponse, strict=False)

def execute_rag_query(user_prompt: str, username: str, past_conversation: List[ChatMessage]) -> FinalResponse:
  print(f"Executing Prompt: {user_prompt} for user : {username}")  
  message_history = get_message_history(past_conversation)
  node1_output = node1_chain.invoke({"query": user_prompt, "history": message_history})
  print(f"Node 1 Output: {node1_output}")
  if node1_output.category != "complex":
    return FinalResponse(response=node1_output.response,has_chart=False,chart_data=None)
  
  print("Executing node 2")
  node2_output = node2.invoke({"query": user_prompt, "pid": username, "history": message_history })
  print(f"\n\nNode 2 Output: {node2_output}\n\n")
  raw_data = node2_output['result']
  
  if not raw_data:
    return FinalResponse(response="Could not find any data relevant to your query. Please ask another question",has_chart=False,chart_data=None)
  
  node3_output = node3_chain.invoke({"query": user_prompt, "raw_data": raw_data})
  print(f"Node 3 Output: {node3_output}")
  return node3_output