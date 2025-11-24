"""
rag.py
Handles:
- Vector DB initialization (Chroma)
- Embedding generation (OpenAI)
- API spec ingestion (add/update)
- Semantic similarity search
"""

import os
import json
import chromadb
from openai import OpenAI
from typing import List, Dict, Any

# -----------------------------
# 1. Initialize OpenAI + Chroma
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# NEW Chroma cient
chroma_client = chromadb.PersistentClient(
    path="./vectorstore/chroma",
    settings=chromadb.Settings(anonymized_telemetry=False)
)

# NEW collection creation
collection = chroma_client.get_or_create_collection("api_specs")


# -----------------------------
# 2. Embedding Helper
# -----------------------------

def embed_text(text: str) -> List[float]:
    """
    Generates an embedding using OpenAI text-embedding-3-large.
    """
    response = client_openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


# -----------------------------
# 3. Ingestion: Add API Specs
# -----------------------------

def ingest_api_spec(api_name, description, input_schema, output_schema):
    # Convert dict → JSON string (allowed by Chroma)
    # 1. Embed DESCRIPTION
    collection.add(
        ids=[api_name + "_desc"],
        embeddings=[embed_text(description)],
        documents=[description],
        metadatas=[{
            "api_name": api_name,
            "type": "description"
        }]
    )
    
    # 2. Embed INPUT SCHEMA ONLY
    collection.add(
        ids=[api_name + "_input"],
        embeddings=[embed_text(json.dumps(input_schema))],
        documents=[json.dumps(input_schema)],
        metadatas=[{
            "api_name": api_name,
            "type": "input_schema"
        }]
    )
    
    # 3. Embed OUTPUT SCHEMA ONLY
    collection.add(
        ids=[api_name + "_output"],
        embeddings=[embed_text(json.dumps(output_schema))],
        documents=[json.dumps(output_schema)],
        metadatas=[{
            "api_name": api_name,
            "type": "output_schema"
        }]
    )

    print(f"🔥 Ingested 3 embeddings for: {api_name}")

    """
    meta = {
        "api_name": api_name,
        "description": description,
        "input_schema": json.dumps(input_schema),
        "output_schema": json.dumps(output_schema)
    }

    # The document text used for embedding
    document_text = (
        f"API: {api_name}\n"
        f"Description: {description}\n"
        f"Input Schema: {json.dumps(input_schema)}\n"
        f"Output Schema: {json.dumps(output_schema)}"
    )

    embedding = embed_text(document_text)

    collection.add(
        ids=[api_name],
        embeddings=[embedding],
        documents=[document_text],
        metadatas=[meta]
    )



    print(f"🔍 Ingested API → {api_name}")
"""

# -----------------------------
# 4. Bulk Ingestion
# -----------------------------

def ingest_api_folder(folder_path: str = "./data/apis"):
    """
    Reads all JSON files in data/apis/ and stores them in ChromaDB.
    Each file should contain: {name, description, input_schema, output_schema}
    """
    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)

        ingest_api_spec(
            api_name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {})
        )


# -----------------------------
# 5. Semantic Search (RAG)
# -----------------------------

def retrieve_similar_api(output_schema: dict, k: int = 1):
    """
    Takes a generated output schema → finds the closest matching API.
    """

    search_text = json.dumps(output_schema)
 
    query_embedding = embed_text(search_text)


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"type": "output_schema"}
    )

    print (results)
    if len(results["ids"][0]) == 0:
        return None

    top_id = results["ids"][0][0]
    top_metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]

    result_obj= {
        "name": top_metadata["api_name"],
        "description": results["documents"][0] if top_metadata["type"] == "description" else None,
        "input_schema": results["documents"][0] if top_metadata["type"] == "input_schema" else None,
        "output_schema": results["documents"][0] if top_metadata["type"] == "output_schema" else None,
        #"input_schema": top_metadata["input_schema"],
        #"output_schema": top_metadata["output_schema"],
        "distance": distance
    }

    # Convert to JSON-safe types (avoid numpy)
    #safe_result = make_json_safe(result_obj)

    print("\n--- MATCHED API (PRINTED IN TERMINAL) ---")
    print(json.dumps(result_obj, indent=2))

    return result_obj


# -----------------------------
# 6. Utility: Reset Vector Store
# -----------------------------
def print_api(api_name):
    data = collection.get()
    for i, meta in enumerate(data["metadatas"]):
        if meta["api_name"] == api_name:
            print(f"\nID: {data['ids'][i]}")
            print(f"Type: {meta['type']}")
            print(f"Document: {data['documents'][i][:200]}...")

def clear_all():
    """
    Deletes the entire Chroma collection.
    """
    chroma_client.delete_collection("api_specs")
    print("🗑️ Cleared all vector data.")


# -----------------------------
# 7. Utility: List Loaded APIs
# -----------------------------

def list_ingested_apis() -> List[str]:
    """
    Returns all API IDs stored in the vector DB.
    """
    try:
        return collection.get()["ids"]
    except Exception:
        return []


# -----------------------------
# Quick Manual Test
# -----------------------------
if __name__ == "__main__":
    print("Current APIs in vector DB:", list_ingested_apis())
