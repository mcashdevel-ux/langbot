#!/usr/bin/env python3
"""
Database exploration script for agent_checkpoints.db
Explores the SQLite database to understand its structure and contents.
"""

import sqlite3
import json
import base64
from datetime import datetime
from collections import defaultdict

DB_PATH = "memory/agent_checkpoints.db"

def connect():
    return sqlite3.connect(DB_PATH)

def explore_structure():
    """Explore database structure."""
    print("=" * 60)
    print("DATABASE STRUCTURE")
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
    
    # Get triggers
    cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='trigger'")
    triggers = cursor.fetchall()
    if triggers:
        print(f"\nTriggers: {triggers}")
    
    conn.close()

def explore_tables():
    """Explore table schemas and sample data."""
    print("\n" + "=" * 60)
    print("TABLE EXPLORATION")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    for table in ['checkpoints', 'writes']:
        print(f"\n{'='*40}")
        print(f"TABLE: {table}")
        print(f"{'='*40}")
        
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        print(f"\nColumns ({len(columns)}):")
        for col in columns:
            cid, name, dtype, notnull, dflt, pk = col
            pk_str = "PK" if pk else ""
            notnull_str = "NOT NULL" if notnull else ""
            dflt_str = f"DEFAULT {dflt}" if dflt else ""
            print(f"  {name}: {dtype} {pk_str} {notnull_str} {dflt_str}")
        
        # Row counts
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"\nTotal rows: {count}")
        
        # Sample data
        print(f"\nSample rows (first 5):")
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cursor.fetchall()
        for row in rows:
            print(f"  {row}")
        
        # Unique values for key columns
        print(f"\nUnique values in key columns:")
        for col in ['thread_id', 'checkpoint_ns', 'checkpoint_id', 'type']:
            cursor.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
            distinct_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT {col} FROM {table} LIMIT 5")
            samples = [str(r[0]) for r in cursor.fetchall()]
            print(f"  {col}: {distinct_count} unique values, samples: {samples}")
    
    conn.close()

def explore_checkpoints():
    """Deep dive into checkpoints table."""
    print("\n" + "=" * 60)
    print("CHECKPOINTS ANALYSIS")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Checkpoint types
    cursor.execute("SELECT type, COUNT(*) FROM checkpoints GROUP BY type")
    print("\nCheckpoint types:")
    for type_, count in cursor.fetchall():
        print(f"  {type_}: {count}")
    
    # Checkpoint namespaces
    cursor.execute("SELECT checkpoint_ns, COUNT(*) FROM checkpoints GROUP BY checkpoint_ns ORDER BY COUNT(*) DESC")
    print("\nTop namespaces:")
    for ns, count in cursor.fetchall()[:10]:
        print(f"  {ns}: {count}")
    
    # Thread distribution
    cursor.execute("SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY COUNT(*) DESC")
    print("\nTop threads:")
    for thread_id, count in cursor.fetchall()[:10]:
        print(f"  {thread_id}: {count}")
    
    # Parent-child relationships
    cursor.execute("SELECT COUNT(*) FROM checkpoints WHERE parent_checkpoint_id IS NOT NULL")
    print(f"\nCheckpoints with parents: {cursor.fetchone()[0]}")
    
    # Sample checkpoint content
    cursor.execute("SELECT checkpoint_id, type, length(checkpoint) FROM checkpoints LIMIT 3")
    print("\nSample checkpoint IDs and sizes:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: type={row[1]}, size={row[2]} bytes")
    
    conn.close()

def explore_writes():
    """Deep dive into writes table."""
    print("\n" + "=" * 60)
    print("WRITES ANALYSIS")
    print("=" * 60)
    
    conn = connect()
    cursor = conn.cursor()
    
    # Write types
    cursor.execute("SELECT type, COUNT(*) FROM writes GROUP BY type")
    print("\nWrite types:")
    for type_, count in cursor.fetchall():
        print(f"  {type_}: {count}")
    
    # Channel distribution
    cursor.execute("SELECT channel, COUNT(*) FROM writes GROUP BY channel ORDER BY COUNT(*) DESC")
    print("\nTop channels:")
    for channel, count in cursor.fetchall()[:10]:
        print(f"  {channel}: {count}")
    
    # Thread distribution
    cursor.execute("SELECT thread_id, COUNT(*) FROM writes GROUP BY thread_id ORDER BY COUNT(*) DESC")
    print("\nTop threads:")
    for thread_id, count in cursor.fetchall()[:10]:
        print(f"  {thread_id}: {count}")
    
    # Task distribution
    cursor.execute("SELECT task_id, COUNT(*) FROM writes GROUP BY task_id ORDER BY COUNT(*) DESC")
    print("\nTop tasks:")
    for task_id, count in cursor.fetchall()[:10]:
        print(f"  {task_id}: {count}")
    
    # Sample write values
    cursor.execute("SELECT checkpoint_id, channel, type, length(value) FROM writes LIMIT 5")
    print("\nSample writes:")
    for row in cursor.fetchall():
        print(f"  checkpoint_id={row[0]}, channel={row[1]}, type={row[2]}, size={row[3]} bytes")
    
    conn.close()

def decode_and_show_checkpoint(checkpoint_id):
    """Decode and display a checkpoint's content."""
    conn = connect()
    cursor = conn.cursor()
    
    cursor.execute("SELECT checkpoint, type FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
    row = cursor.fetchone()
    
    if not row:
        print(f"Checkpoint {checkpoint_id} not found")
        conn.close()
        return
    
    checkpoint_data = row[0]
    checkpoint_type = row[1]
    
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {checkpoint_id}")
    print(f"{'='*60}")
    print(f"Type: {checkpoint_type}")
    print(f"Size: {len(checkpoint_data)} bytes")
    
    try:
        if checkpoint_type == 'json':
            decoded = json.loads(checkpoint_data)
            print(f"Content: {json.dumps(decoded, indent=2)}")
        elif checkpoint_type == 'text':
            decoded = checkpoint_data.decode('utf-8')
            print(f"Content: {decoded[:500]}...")
        elif checkpoint_type == 'pickle':
            import pickle
            decoded = pickle.loads(checkpoint_data)
            print(f"Content type: {type(decoded)}")
            print(f"Content: {str(decoded)[:500]}...")
        else:
            print(f"Unknown type: {checkpoint_type}")
            print(f"Raw (first 200 bytes): {checkpoint_data[:200]}")
    except Exception as e:
        print(f"Error decoding: {e}")
    
    conn.close()

def main():
    """Run all explorations."""
    print("\n" + "=" * 60)
    print("AGENT CHECKPOINTS DATABASE EXPLORATION")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Exploration time: {datetime.now()}")
    
    explore_structure()
    explore_tables()
    explore_checkpoints()
    explore_writes()
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print("\nTo decode a specific checkpoint, run:")
    print(f"  python explore_db.py --show <checkpoint_id>")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        if len(sys.argv) > 2:
            decode_and_show_checkpoint(sys.argv[2])
        else:
            print("Usage: python explore_db.py --show <checkpoint_id>")
    else:
        main()
