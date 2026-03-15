"""
Script to inspect MongoDB structure before adding embeddings.
This helps verify what collections exist and their document structure.
"""

import os
from pymongo import MongoClient
import json

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "banking_db"

def inspect_database():
    """Inspect MongoDB collections and sample documents."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {MONGO_URI}")
        
        db = client[DB_NAME]
        
        # List all collections
        collections = db.list_collection_names()
        print(f"\n{'='*60}")
        print(f"Database: {DB_NAME}")
        print(f"Collections found: {len(collections)}")
        print(f"{'='*60}\n")
        
        for coll_name in sorted(collections):
            collection = db[coll_name]
            
            # Count documents
            total_docs = collection.count_documents({})
            docs_with_embedding = collection.count_documents({"embedding": {"$exists": True}})
            docs_without_embedding = total_docs - docs_with_embedding
            
            print(f"\n📊 Collection: {coll_name}")
            print(f"   Total documents: {total_docs}")
            print(f"   With embeddings: {docs_with_embedding}")
            print(f"   Without embeddings: {docs_without_embedding}")
            
            if total_docs > 0:
                # Get sample document
                sample = collection.find_one()
                
                # Show structure (field names only, not full data)
                print(f"   Fields: {list(sample.keys())}")
                
                # Show one example document structure
                print(f"\n   Sample document structure:")
                for key, value in list(sample.items())[:10]:  # First 10 fields
                    if key == "_id":
                        print(f"      {key}: ObjectId(...)")
                    elif key == "embedding":
                        print(f"      {key}: [array of {len(value)} numbers]")
                    elif isinstance(value, dict):
                        print(f"      {key}: {{...}} (dict with {len(value)} keys)")
                    elif isinstance(value, list):
                        print(f"      {key}: [...] (list with {len(value)} items)")
                    else:
                        print(f"      {key}: {type(value).__name__}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Is MongoDB running? Try: brew services list | grep mongodb")
        print(f"2. Check connection string: {MONGO_URI}")
        return False
    
    return True

if __name__ == "__main__":
    inspect_database()
