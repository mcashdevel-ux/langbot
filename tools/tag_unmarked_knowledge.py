#!/usr/bin/env python3
"""Script to tag untagged knowledge entries in Supabase and sync them locally.

Uses the local sentence-transformers embedding model (all-MiniLM-L6-v2)
to perform fast zero-shot tag classification via cosine similarity.
"""

import os
import sys
import json
import time
import math
import requests
from typing import List, Dict, Any

# Add current directory to path so components can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.vault import bootstrap as _vault_bootstrap
from components.config import config
from components import memory_store

# Colors for terminal printing
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {msg}")

def log_error(msg: str):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)

# Predefined tag candidates representing common topics in the knowledge base
CANDIDATE_TAGS = [
    "bug-fix", "model-names", "credentials", "database", "git", "cli",
    "system-cleanup", "python", "javascript", "docker", "gemma", "qwen",
    "openai", "supabase", "memory", "config", "installation", "documentation",
    "testing", "benchmarks", "web-search", "api-key", "performance",
    "deployment", "script", "agent", "process-management", "error-handling"
]

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}=== Langbot Knowledge Tagging Script (ANN-based) ==={Colors.ENDC}\n")
    
    # 1. Bootstrap Vault and credentials
    _vault_bootstrap()
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    
    if not url or not key:
        log_error("Supabase credentials missing in env or vault. Please configure SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        sys.exit(1)
        
    # 2. Load Embedding model
    log_info("Warming up local embedding model...")
    try:
        embedder = memory_store.get_embeddings(announce=False)
        log_success("Embedding model ready.")
    except Exception as e:
        log_error(f"Failed to load embedding model: {e}")
        sys.exit(1)
        
    # Pre-embed the candidate tags
    log_info(f"Computing embeddings for {len(CANDIDATE_TAGS)} candidate tags...")
    tag_vectors = embedder.embed_documents(CANDIDATE_TAGS)
    
    # 3. Retrieve untagged records from Supabase
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    
    log_info("Fetching all knowledge entries from Supabase (paginated)...")
    records = []
    page_size = 1000
    offset = 0
    try:
        while True:
            page_headers = dict(headers)
            page_headers["Range-Unit"] = "items"
            page_headers["Range"] = f"{offset}-{offset + page_size - 1}"
            r = requests.get(
                f"{url}/rest/v1/knowledge",
                headers=page_headers,
                params={"select": "id,fact,tags", "stale": "eq.false", "order": "id.asc"},
                timeout=30
            )
            if r.status_code not in (200, 206):
                log_error(f"Failed to fetch records: HTTP {r.status_code} - {r.text}")
                sys.exit(1)
            chunk = r.json()
            if not chunk:
                break
            records.extend(chunk)
            log_info(f"  Fetched {len(records)} records so far...")
            if len(chunk) < page_size:
                break
            offset += page_size
    except Exception as e:
        log_error(f"Error connecting to Supabase: {e}")
        sys.exit(1)
        
    # Filter untagged or empty tags
    untagged_records = [r for r in records if not r.get("tags") or len(r["tags"]) == 0]
    total_untagged = len(untagged_records)
    
    if total_untagged == 0:
        log_success("All records in Supabase are already tagged! Nothing to do.")
        sys.exit(0)
        
    log_info(f"Found {len(records)} total records. {total_untagged} are untagged.")
    
    # 4. Initialize ChromaDB
    chroma_collection = None
    try:
        chroma_collection = memory_store.get_collection()
        log_info(f"Local ChromaDB connection active. {chroma_collection.count()} local facts loaded.")
    except Exception as e:
        log_warning(f"Could not open local ChromaDB collection: {e}. Local updates will be skipped.")
        
    # 5. Process tagging via ANN zero-shot classification
    updated_supabase = 0
    updated_chroma = 0
    
    # Batch retrieve embeddings of the facts for performance
    batch_size = 100
    log_info(f"Classifying tags in batches of {batch_size}...")
    
    for i in range(0, total_untagged, batch_size):
        batch = untagged_records[i:i + batch_size]
        facts_text = [item["fact"] for item in batch]
        
        # Embed the facts in batch
        try:
            fact_vectors = embedder.embed_documents(facts_text)
        except Exception as e:
            log_error(f"Failed to embed batch: {e}")
            continue
            
        for idx, item in enumerate(batch):
            item_id = item["id"]
            fact = item["fact"]
            fact_vector = fact_vectors[idx]
            
            # Compute similarity to all candidate tags
            assigned_tags = []
            for tag, tag_vector in zip(CANDIDATE_TAGS, tag_vectors):
                sim = cosine_similarity(fact_vector, tag_vector)
                if sim >= 0.35:  # Similarity threshold
                    assigned_tags.append(tag)
                    
            # Incorporate regex-based helper auto_tags
            assigned_tags.extend(memory_store.auto_tags(fact))
            
            # Clean and clean duplicates/limit
            tags = memory_store.clean_tags(assigned_tags)
            if not tags:
                tags = ["general"]
                
            # 5a. Update Supabase
            try:
                patch_r = requests.patch(
                    f"{url}/rest/v1/knowledge?id=eq.{item_id}",
                    headers=headers,
                    json={"tags": tags},
                    timeout=10
                )
                if patch_r.status_code in (200, 204):
                    updated_supabase += 1
                else:
                    log_error(f"Failed to update Supabase record {item_id}: HTTP {patch_r.status_code}")
            except Exception as e:
                log_error(f"Error updating Supabase record {item_id}: {e}")
                
            # 5b. Update ChromaDB
            if chroma_collection:
                try:
                    norm = memory_store.normalize(fact)
                    exact = chroma_collection.get(
                        where={"norm": norm},
                        limit=1,
                        include=["metadatas", "embeddings"]
                    )
                    if exact and exact.get("ids"):
                        mem_id = exact["ids"][0]
                        meta = dict(exact["metadatas"][0])
                        meta["tags"] = ",".join(tags)
                        document = norm + "".join(f" #{t}" for t in tags)
                        embeds = exact.get("embeddings")
                        vector = memory_store._vector(embeds[0]) if embeds and len(embeds) else None
                        
                        chroma_collection.update(
                            ids=[mem_id],
                            embeddings=[vector] if vector is not None else None,
                            metadatas=[meta],
                            documents=[document]
                        )
                        updated_chroma += 1
                except Exception as e:
                    log_warning(f"Error updating local ChromaDB for record {item_id}: {e}")
                    
        log_info(f"Progress: {i + len(batch)}/{total_untagged} items processed.")
        
    log_success(f"Successfully processed all {total_untagged} entries.")
    log_success(f"Updated {updated_supabase} entries in Supabase.")
    log_success(f"Updated {updated_chroma} matching entries in local ChromaDB.")

if __name__ == "__main__":
    main()
