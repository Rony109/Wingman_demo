
import os
import faiss
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from new_raw import data_catalog  # Must be nested structure: databases > tables > fields
from together import Together


load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct-Lite")
INDEX_PATH = os.getenv("INDEX_PATH", "catalog_index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "catalog_metadata.pkl")



def load_catalog_assistant(index_path=INDEX_PATH, metadata_path=METADATA_PATH, force_retrain=False):
    """
    Loads FAISS index and metadata if available.
    Trains embeddings and builds index only if files are missing or force_retrain is True.
    """
    if not force_retrain and os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        model = SentenceTransformer(metadata["model_name"])
        return model, index, metadata["catalog_metadata"], metadata["catalog_texts"]


def search_catalog(query, model, index, catalog_metadata, catalog_texts, top_k=25):
    print(f"\n🔍 Searching for: '{query}'")
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        results.append({
            "rank": rank,
            "distance": dist,
            "metadata": catalog_metadata[idx],
            "text": catalog_texts[idx],
            "similarity_score": 1 / (1 + dist)
        })
    return results


def display_results(results):
    """
    Groups search results by database and table,
    and prints the matched field names (columns) grouped under each table.
    """
    from collections import defaultdict

    # Nested dict: db_name -> table_name -> list of field names
    grouped = defaultdict(lambda: defaultdict(list))

    for r in results:
        md = r["metadata"]
        db = md["database_name"]
        table = md["table_name"]
        field = md["field_name"]
        grouped[db][table].append(field)

    for db_name, tables in grouped.items():

        for table_name, fields in tables.items():
            # Join fields with commas
            fields_str = ", ".join(fields)


client = Together(api_key=TOGETHER_API_KEY)

def query_llama_together(prompt):
    """
    Sends a structured prompt to Together's LLaMA model and returns the explanation.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()



def single_query_search(query):

    model, index, catalog_metadata, catalog_texts = load_catalog_assistant()

    results = search_catalog(query, model, index, catalog_metadata, catalog_texts)

    # Build prompt for Hugging Face explanation
    max_entries = 5
    max_chars = 1500
    total_len = 0
    context_parts = []

    for r in results[:max_entries]:
        md = r["metadata"]
        db_name = md.get("database_name", "Unknown DB")
        db_desc = md.get("database_description", "")
        table_name = md.get("table_name", "Unknown Table")
        table_desc = md.get("table_description", "")
        field_name = md["field_name"]
        field_desc = md["business_description"]
        tags = ", ".join(md.get("tags", []))
        
        part = (
            f"Database: {db_name} - {db_desc}\n"
            f"Table: {table_name} - {table_desc}\n"
            f"Field: {field_name}\n"
            f"Description: {field_desc}\n"
        )
        if tags:
            part += f"Tags: {tags}\n"

        if total_len + len(part) > max_chars:
            break
        context_parts.append(part)
        total_len += len(part)

    prompt = (
        "You are a helpful assistant explaining database metadata fields.\n"
        "Given these field descriptions:\n\n" +
        "\n---\n".join(context_parts) +
        f"\n\nPlease provide a concise, clear explanation related to the query: '{query}'."
    )

    explanation = query_llama_together(prompt)
    print(explanation)
    return explanation


