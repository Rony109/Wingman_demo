# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import faiss
import sys
import pickle
import os
from sentence_transformers import SentenceTransformer
from raw_data import data_catalog

# COMMAND ----------

# Paths
INDEX_PATH = "catalog_index.faiss"
METADATA_PATH = "catalog_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# COMMAND ----------

# Import the actual data
from raw_data import data_catalog  # raw_data.py must define `data_catalog` as a Python list

# COMMAND ----------

def create_catalog_texts(data):
    catalog_texts = []
    catalog_metadata = []

    for database in data:
        for table in database['tables']:
<<<<<<< HEAD
            for field in table['fields']:  # 👈 renamed from 'columns' to 'fields'
                tags = field.get('tags', [])
                pii_note = "This field contains PII data." if "PII" in tags else ""
                sample_values = ", ".join(str(val) for val in field.get('sample_values', [])[:2])
=======
            for column in table['fields']:
                tags = column.get('tags', [])
                pii_note = "This field contains PII data." if "PII" in tags else ""
                sample_values = ", ".join([str(sample_value) for sample_value in column.get('sample_values', [])][:2])
>>>>>>> 574f643bc9cbfc420fa41cf03eaf612778b56b7a


                # Create a descriptive natural-language text for semantic embedding
                text = (
<<<<<<< HEAD
                    f"The field '{field['field_name']}' (Business name: {field['business_name']}) "
                    f"in table '{table['table_name']}' from database '{database['database_code']}' "
                    f"captures the following: {field['business_description']}. "
                    f"It is a {field['data_type']} field. {pii_note} "
=======
                    f"The field '{column['field_name']}' (Business name: {column['business_name']}) "
                    f"in table '{table['table_name']}' from database '{database['database_code']}' "
                    f"captures the following: {column['business_description']}. "
                    f"It is a {column['data_type']} field. {pii_note} "
>>>>>>> 574f643bc9cbfc420fa41cf03eaf612778b56b7a
                    f"Tags include: {', '.join(tags)}. "
                    f"Sample values are: {sample_values}."
                ).strip()

                catalog_texts.append(text)

                metadata = {
                    'database_name': database['database_code'],
                    'database_description': database['database_description'],
                    'table_name': table['table_name'],
                    'table_description': table['table_description'],
<<<<<<< HEAD
                    'field_name': field['field_name'],
                    'business_name': field['business_name'],
                    'business_description': field['business_description'],
                    'data_type': field['data_type'],
                    'length': field.get('length'),
                    'tags': field.get('tags', []),
                    'sample_values': field.get('sample_values', [])
=======
                    'field_name': column['field_name'],
                    'business_name': column['business_name'],
                    'business_description': column['business_description'],
                    'data_type': column['data_type'],
                    'length': column.get('length'),
                    'tags': column.get('tags', []),
                    'sample_values': [str(sample_value) for sample_value in column.get('sample_values', [])]
>>>>>>> 574f643bc9cbfc420fa41cf03eaf612778b56b7a
                }
                catalog_metadata.append(metadata)

    return catalog_texts, catalog_metadata

# COMMAND ----------

def deduplicate_catalog_entries(catalog_texts, catalog_metadata):
    from collections import defaultdict

    seen_fields = set()
    dedup_texts = []
    dedup_metadata = []

    for text, meta in zip(catalog_texts, catalog_metadata):
        key = (meta['database_name'], meta['table_name'], meta['field_name'])
        if key not in seen_fields:
            seen_fields.add(key)
            dedup_texts.append(text)
            dedup_metadata.append(meta)

    return dedup_texts, dedup_metadata

# COMMAND ----------

def main():
    # # Load database metadata
    # print("Loading database metadata...")
    # data = json.loads(json_data)
    
    # Create catalog texts for embedding
    print("Creating catalog texts...")
    catalog_texts, catalog_metadata = create_catalog_texts(data_catalog)

    # ✅ Deduplicate
    catalog_texts, catalog_metadata = deduplicate_catalog_entries(catalog_texts, catalog_metadata)
    print(f"✅ Deduplicated to {len(catalog_texts)} unique fields")
    
    print(f"Generated {len(catalog_texts)} field descriptions")
    
    # Load model and generate embeddings
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings...")
    embeddings = model.encode(catalog_texts, convert_to_numpy=True)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Build and save FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    
    # Save metadata
    print("Saving metadata...")
    metadata_to_save = {
        "catalog_metadata": catalog_metadata,
        "catalog_texts": catalog_texts,
        "model_name": MODEL_NAME
    }
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_to_save, f)
    
    print("✅ Model and index saved successfully!")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    
    # Test the index with a sample query
    print("\n🧪 Testing with sample queries...")
    test_queries = [
        "customer name",
        "account number", 
        "phone number",
        "transaction date"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search in index
        distances, indices = index.search(query_embedding, k=3)
        
        print("Top 3 matches:")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            metadata = catalog_metadata[idx]
            print(f"  {i+1}. {metadata['database_name']}.{metadata['table_name']}.{metadata['field_name']} - {metadata['business_name']} (distance: {distance:.3f})")

# COMMAND ----------

if __name__ == "__main__":
    main()
