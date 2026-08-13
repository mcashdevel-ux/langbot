#!/usr/bin/env python3
"""
Chroma database exploration script.
Explores the Chroma vector database SQLite storage.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "memory/agent_memory_chroma/chroma.sqlite3"

def connect():
    return sqlite3.connect(DB_PATH)

def explore_structure():
    """Explore database structure."""
    print("=" * 60)
    print("CHROMA DATABASE STRUCTURE")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nTables found: {[t[0] for t in tables]}")
    
    # Get indexes
    cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'")
    indexes = cursor.fetchall()
    if indexes:
        print(f"\nIndexes: {indexes}")
    
    conn.close()

def explore_collections():
    """Explore collections and their metadata."""
    print("\n" + "=" * 60)
    print("COLLECTIONS")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Get collections
    cursor.execute("SELECT * FROM collections")
    collections = cursor.fetchall()
    print(f"\nCollections ({len(collections)}):")
    for row in collections:
        print(f"  id={row[0]}, name={row[1]}, dimension={row[2]}, database_id={row[3]}")
    
    # Get collection metadata
    cursor.execute("SELECT * FROM collection_metadata")
    meta = cursor.fetchall()
    print(f"\nCollection metadata ({len(meta)}):")
    for row in meta:
        print(f"  {row}")
    
    # Get segments
    cursor.execute("SELECT * FROM segments")
    segments = cursor.fetchall()
    print(f"\nSegments ({len(segments)}):")
    for row in segments:
        print(f"  id={row[0]}, type={row[1]}, scope={row[2]}, collection={row[3]}")
    
    # Get segment metadata
    cursor.execute("SELECT * FROM segment_metadata")
    seg_meta = cursor.fetchall()
    print(f"\nSegment metadata ({len(seg_meta)}):")
    for row in seg_meta:
        print(f"  {row}")
    
    conn.close()

def explore_embeddings():
    """Explore embeddings."""
    print("\n" + "=" * 60)
    print("EMBEDDINGS")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Count
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    count = cursor.fetchone()[0]
    print(f"\nTotal embeddings: {count}")
    
    # Sample embeddings
    cursor.execute("SELECT id, segment_id, embedding_id, seq_id, created_at FROM embeddings LIMIT 5")
    print("\nSample embeddings:")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    # Embedding metadata
    cursor.execute("SELECT key, string_value FROM embedding_metadata")
    print(f"\nEmbedding metadata ({len(cursor.fetchall())}):")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()

def explore_embeddings_queue():
    """Explore embeddings queue."""
    print("\n" + "=" * 60)
    print("EMBEDDINGS QUEUE")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Count
    cursor.execute("SELECT COUNT(*) FROM embeddings_queue")
    count = cursor.fetchone()[0]
    print(f"\nTotal queued embeddings: {count}")
    
    # Sample queue items
    cursor.execute("SELECT * FROM embeddings_queue LIMIT 5")
    print("\nSample queue items:")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()

def explore_fulltext_search():
    """Explore fulltext search tables."""
    print("\n" + "=" * 60)
    print("FULLTEXT SEARCH")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    tables = ['embedding_fulltext_search', 'embedding_fulltext_search_data', 
              'embedding_fulltext_search_idx', 'embedding_fulltext_search_content',
              'embedding_fulltext_search_docsize', 'embedding_fulltext_search_config']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"\n{table}: {count} rows")
    
    # Show config
    cursor.execute("SELECT * FROM embedding_fulltext_search_config")
    print("\nFulltext search config:")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()

def explore_tenants_databases():
    """Explore tenants and databases."""
    print("\n" + "=" * 60)
    print("TENANTS AND DATABASES")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Tenants
    cursor.execute("SELECT * FROM tenants")
    tenants = cursor.fetchall()
    print(f"\nTenants ({len(tenants)}):")
    for row in tenants:
        print(f"  {row}")
    
    # Databases
    cursor.execute("SELECT * FROM databases")
    databases = cursor.fetchall()
    print(f"\nDatabases ({len(databases)}):")
    for row in databases:
        print(f"  {row}")
    
    conn.close()

def explore_migrations():
    """Explore migrations."""
    print("\n" + "=" * 60)
    print("MIGRATIONS")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM migrations")
    migrations = cursor.fetchall()
    print(f"\nMigrations ({len(migrations)}):")
    for row in migrations:
        print(f"  dir={row[0]}, version={row[1]}, filename={row[2]}")
    
    conn.close()

def main():
    """Run all explorations."""
    print("\n" + "=" * 60)
    print("CHROMA DATABASE EXPLORATION")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Exploration time: {datetime.now()}")
    
    explore_structure()
    explore_collections()
    explore_embeddings()
    explore_embeddings_queue()
    explore_fulltext_search()
    explore_tenants_databases()
    explore_migrations()
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
