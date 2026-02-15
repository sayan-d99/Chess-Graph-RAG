from neo4j import GraphDatabase, Result
import streamlit as st

@st.cache_resource()
def get_db_driver():
  driver = GraphDatabase.driver(
    uri=st.secrets.NEO4J_URI, 
    auth=(st.secrets.NEO4J_USERNAME, st.secrets.NEO4J_PASSWORD)
  )
  driver.verify_connectivity()
  return driver

db_driver = get_db_driver()

def execute_db_query(query, params, result_transformer=Result.to_eager_result):
  query_result = None
  try:
    query_result = db_driver.execute_query(query, params, result_transformer_=result_transformer)
  except Exception as e:
    print(f"Exception when executing graph query: {e}")
  else:
    return query_result

