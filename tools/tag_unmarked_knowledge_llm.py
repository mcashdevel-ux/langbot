#!/usr/bin/env python3
"""Script to tag untagged knowledge entries in Supabase and sync them locally.

Uses the configured local OpenAI-compatible LLM to generate tags in batches.
Updates both Supabase and the local ChromaDB long-term memory.
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Any

# Add current directory to path so components can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.vault import bootstrap as _vault_bootstrap
from components.config import config
from components import memory_store
from langchain_openai import ChatOpenAI

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

def clean_json_content(content: str) -> str:
    """Strip markdown formatting if present."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()

def tag_batch(llm: ChatOpenAI, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Send a batch of facts to the local model to generate tags."""
    prompt = f"""You are a tagging assistant.
Analyze the following list of facts/knowledges. For each fact, produce a list of 1 to 5 concise, lowercase, alphanumeric-and-hyphen-only tags.
Return ONLY a valid JSON list of objects, each containing "id" and "tags". Do not include any explanations, markdown headers, or other text.

Example format:
[
  {{"id": 1, "tags": ["tag-a", "tag-b"]}},
  {{"id": 2, "tags": ["tag-c"]}}
]

Facts to tag:
{json.dumps([{"id": item["id"], "fact": item["fact"]} for item in batch], indent=2)}
"""
    try:
        response = llm.invoke(prompt)
        cleaned_content = clean_json_content(response.content)
        result = json.loads(cleaned_content)
        if isinstance(result, list):
            return result
        log_error(f"LLM did not return a list: {cleaned_content}")
    except Exception as e:
        log_error(f"Error invoking LLM or parsing response: {e}")
    return []

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}=== Langbot Knowledge Tagging Script (LLM-based) ==={Colors.ENDC}\n")
    
    # 1. Bootstrap Vault and credentials
    _vault_bootstrap()
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    
    if not url or not key:
        log_error("Supabase credentials missing in env or vault. Please configure SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        sys.exit(1)
        
    # 2. Configure LLM
    base_url = config.get("llm.base_url", "http://127.0.0.1:8080/v1")
    model = config.get("llm.model", "local-model")
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="not-needed",
        temperature=0.0,
    )
    
    log_info(f"Targeting local LLM at {base_url} (model: {model})")
    
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
        
    # 5. Process in batches
    batch_size = 5
    processed_count = 0
    updated_supabase = 0
    updated_chroma = 0
    
    for i in range(0, total_untagged, batch_size):
        batch = untagged_records[i:i + batch_size]
        log_info(f"Tagging batch {i//batch_size + 1} ({len(batch)} items, progress: {processed_count}/{total_untagged})...")
        
        tagged_results = tag_batch(llm, batch)
        if not tagged_results:
            log_warning("Batch tagging failed. Retrying batch individually...")
            tagged_results = []
            for item in batch:
                res = tag_batch(llm, [item])
                if res:
                    tagged_results.extend(res)
                    
        # Create a lookup for tags
        tag_map = {}
        for res in tagged_results:
            if "id" in res and "tags" in res:
                tag_map[res["id"]] = memory_store.clean_tags(res["tags"])
                
        # Apply updates
        for item in batch:
            item_id = item["id"]
            fact = item["fact"]
            tags = tag_map.get(item_id)
            
            if not tags:
                # Deterministic tags fallback if LLM failed
                tags = memory_store.clean_tags(memory_store.auto_tags(fact))
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
                    
        processed_count += len(batch)
        time.sleep(0.2)  # Short pause to prevent overwhelming the local llama-server
        
    log_success(f"Successfully processed {processed_count} entries.")
    log_success(f"Updated {updated_supabase} entries in Supabase.")
    log_success(f"Updated {updated_chroma} matching entries in local ChromaDB.")

if __name__ == "__main__":
    main()
