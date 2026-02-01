import streamlit as st

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models import Node1ClassificationOutput, FinalResponse
from rag.embedding import ChessLMEmbeddings
from util import load_file

print("Loading prompts")
CYPHER_QUERY_PROMPT = load_file("prompts/cypher_generation.txt")
DECISION_PROMPT = load_file("prompts/decision_prompt.txt")
RESPONSE_GENERATOR_PROMPT = load_file("prompts/response_generator.txt")
print("Prompts loaded")

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

chat_model = get_model_obj()
vector_store, embedding_model, graph = setup_graph_and_vector()

node1_prompt =  ChatPromptTemplate.from_messages([
    ("system", DECISION_PROMPT),
    ("human", "{query}")
])
node1_chain = node1_prompt | chat_model.with_structured_output(Node1ClassificationOutput)

node2_prompt = ChatPromptTemplate([
  ("system", CYPHER_QUERY_PROMPT),
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
  allow_dangerous_requests=True
)

node3_prompt = ChatPromptTemplate.from_messages([
  ("system", RESPONSE_GENERATOR_PROMPT),
  ("human", "Query: {query}\nDatabase Data: {raw_data}")
])
node3_chain = node3_prompt | chat_model.with_structured_output(FinalResponse, strict=False)

def execute_rag_query(user_prompt, username):
  print(f"Executing Prompt: {user_prompt} for user : {username}")
  node1_output = node1_chain.invoke({"query": user_prompt})
  # print(f"Node 1 Output: {node1_output}")
  if node1_output.category != "complex":
    return FinalResponse(response=node1_output.response,has_chart=False,chart_data=None)
  
  node2_output = node2.invoke({"query": user_prompt, "pid": username})
  # print(f"Node 2 Output: {node2_output}")
  raw_data = node2_output['result']
  
  if not raw_data:
    return FinalResponse(response="Could not find any data relevant to your query. Please ask another question",has_chart=False,chart_data=None)
  
  node3_output = node3_chain.invoke({"query": user_prompt, "raw_data": raw_data})
  # print(f"Node 3 Output: {node3_output}")
  return node3_output