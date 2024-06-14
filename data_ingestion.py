from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
import json

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embed_model = ChatGoogleGenerativeAI(model="models/text-embedding-004", temperature=0.8)

# embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

filename = ""
with open(filename, 'r') as f:
    json.load(f)

result = ''

# Recursive function to iterate over the JSON data
def flatten_json(obj, path=''):
    global result
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            flatten_json(value, new_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            new_path = f"{path}[{index}]"
            flatten_json(item, new_path)
    else:
        result += f"{path}: {obj}\n"

flatten_json(data)

print(result)

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
docs = text_splitter.split_documents([result])

qdrant = Qdrant.from_documents(
    docs, 
    embed_model,
    path = "./db",
    collection_name = "medical document"
)

retriever = qdrant.as_retriever()

#ReRanker

compressor = FlashrankRerank(model="ms-macro-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever=retriever)

query = ""

reranked_docs = compression_retriever.invoke(query)

for doc in reranked_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content}\n")
    print(f"score: {doc.metadata['relevance_score']}")
    print("-"*80)
    print("\n\n")