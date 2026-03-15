"""
One-time script to add embeddings to existing MongoDB documents.
This prepares your data for semantic search without changing existing fields.

Usage:
    python add_embeddings.py --collection loan
    python add_embeddings.py --collection fixed_deposits
    python add_embeddings.py --all
"""
# Use for running to add embeddings to existing documents
import os
import sys
import argparse
from typing import List, Dict, Any
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB Configuration (same as main.py)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "banking_db"

# Collections to process - ALL fields for comprehensive semantic search
COLLECTIONS = {
    "savings_accounts": [
        "product_name", "bank_name", "cutoff_date", "description",
        "interest_rate", "product_type", "schema_version", "scraped_at",
        "source_url", "updated_at", "bank_code"
    ],
    "loan": [
        "bank", "product_type", "loan_name", "loan_category", "description",
        "currency", "loan_amount", "loan_term", "interest_rate", "repayment_modes",
        "fees", "ltv_ratio", "insurance_requirements", "acceptable_collateral",
        "borrower_eligibility", "documents_required", "advantages",
        "additional_information", "source_url", "updated_date", "bank_code",
        "schema_version"
    ],
    "credit_cards": [
        "bank", "product_type", "card_name", "network", "tier", "currency",
        "fees", "limits", "interest", "expiry_years", "source_url",
        "updated_at", "bank_code", "schema_version"
    ],
    "fixed_deposits": [
        "bank_name", "product_name", "product_type", "segment",
        "currencies_supported", "interest_payment_options", "minimum_deposit",
        "benefits", "requirements", "term_deposit_types", "notes",
        "source_url", "inserted_at", "bank_code", "schema_version"
    ],
    "exchange_rates": [
        "bank_code", "bank_name", "rate_type", "source_channel", "as_of",
        "base_currency", "rates", "notes", "status", "created_at", "updated_at"
    ],
    "bank_code": [
        "bank_code", "bank_name", "country", "bank_type", "is_active", "created_at"
    ],
    "bank_master": [
        "bank_code", "bank_name", "is_active"
    ],
}


def flatten_interest_rates(doc: Dict[str, Any]) -> str:
    """
    Flatten complex interest rate structure into readable text.
    Handles two different formats:
    1. KB Prasac style: {growth_modes: {maturity_growth: [...], monthly_growth: [...]}}
    2. Wing Bank style: [{term_months: 1, at_maturity: {...}, monthly: {...}}]
    
    Example output:
    "1 month: 1.5% USD 2% KHR, 3 months: 2.75% USD 3.25% KHR, 48 months: 4.75% USD 6% KHR"
    """
    if "interest_rates" not in doc:
        return ""
    
    rates_data = doc["interest_rates"]
    parts = []
    
    # Format 1: Dict with growth_modes (KB Prasac style)
    if isinstance(rates_data, dict) and "growth_modes" in rates_data:
        growth_modes = rates_data.get("growth_modes", {})
        
        # Process each growth mode (maturity_growth, monthly_growth)
        for mode_name, tiers in growth_modes.items():
            if not isinstance(tiers, list):
                continue
            
            mode_label = mode_name.replace("_", " ")
            parts.append(f"{mode_label}:")
            
            # Process each tier
            for tier in tiers:
                if not isinstance(tier, dict) or "months" not in tier:
                    continue
                
                months = tier.get("months", [])
                if not months:
                    continue
                
                # List each month explicitly for better matching
                for month in months:
                    rates = []
                    for currency in ["USD", "KHR"]:
                        if currency in tier:
                            rates.append(f"{tier[currency]}% {currency}")
                    
                    if rates:
                        month_str = f"{month} month" if month == 1 else f"{month} months"
                        parts.append(f"{month_str} {' '.join(rates)}")
    
    # Format 2: List of term entries (Wing Bank style)
    elif isinstance(rates_data, list):
        for entry in rates_data:
            if not isinstance(entry, dict) or "term_months" not in entry:
                continue
            
            term = entry.get("term_months")
            month_str = f"{term} month" if term == 1 else f"{term} months"
            
            # Check both at_maturity and monthly payment modes
            for mode_name, mode_key in [("at maturity", "at_maturity"), ("monthly", "monthly")]:
                if mode_key in entry and entry[mode_key]:
                    mode_rates = entry[mode_key]
                    if isinstance(mode_rates, dict):
                        rates = []
                        for currency in ["USD", "KHR"]:
                            if currency in mode_rates and mode_rates[currency] is not None:
                                rates.append(f"{mode_rates[currency]}% {currency}")
                        
                        if rates:
                            parts.append(f"{month_str} {mode_name}: {' '.join(rates)}")
    
    return " | ".join(parts) if parts else ""


def flatten_loan_rates(doc: Dict[str, Any]) -> str:
    """Flatten loan interest rates into readable text."""
    if "interest_rate" not in doc:
        return ""
    
    rate_data = doc["interest_rate"]
    if isinstance(rate_data, dict):
        parts = []
        if "min_rate" in rate_data or "max_rate" in rate_data:
            min_r = rate_data.get("min_rate", "")
            max_r = rate_data.get("max_rate", "")
            if min_r and max_r:
                parts.append(f"interest rate {min_r}% to {max_r}%")
            elif min_r:
                parts.append(f"interest rate from {min_r}%")
            elif max_r:
                parts.append(f"interest rate up to {max_r}%")
        if "rate" in rate_data:
            parts.append(f"interest rate {rate_data['rate']}%")
        return " ".join(parts)
    return str(rate_data)


def create_searchable_text(doc: Dict[str, Any], fields: List[str], collection_name: str = "") -> str:
    """
    Concatenate relevant fields into a single searchable text string.
    Handles complex nested structures in your banking data.
    
    Args:
        doc: MongoDB document
        fields: List of field names to include
        collection_name: Name of collection (for special handling)
    
    Returns:
        Space-separated string of field values
    """
    parts = []
    
    # Special handling for different collections - flatten complex structures
    if collection_name == "fixed_deposits":
        interest_text = flatten_interest_rates(doc)
        if interest_text:
            parts.append(interest_text)
    elif collection_name == "loan":
        loan_rate_text = flatten_loan_rates(doc)
        if loan_rate_text:
            parts.append(loan_rate_text)
    
    for field in fields:
        value = doc.get(field)
        if value is not None:
            # Handle different data types
            if isinstance(value, list):
                # Lists: join items with spaces
                for item in value:
                    if isinstance(item, dict):
                        # Extract values from nested dicts recursively
                        parts.append(" ".join(str(v) for v in item.values() if v))
                    else:
                        parts.append(str(item))
            elif isinstance(value, dict):
                # Dicts: extract all values recursively
                def extract_dict_values(d):
                    vals = []
                    for k, v in d.items():
                        if isinstance(v, dict):
                            vals.extend(extract_dict_values(v))
                        elif isinstance(v, list):
                            vals.extend([str(item) for item in v if item])
                        elif v:
                            vals.append(str(v))
                    return vals
                parts.extend(extract_dict_values(value))
            else:
                parts.append(str(value).strip())
    
    # Clean and normalize text
    text = " ".join(parts)
    text = " ".join(text.split())  # Remove extra whitespace
    return text


def add_embeddings_to_collection(
    collection_name: str,
    model: SentenceTransformer,
    batch_size: int = 32,
    dry_run: bool = False
):
    """
    Add 'searchable_text' and 'embedding' fields to documents in a collection.
    
    Args:
        collection_name: Name of MongoDB collection
        model: SentenceTransformer model for generating embeddings
        batch_size: Number of documents to process at once
        dry_run: If True, don't actually update the database
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    # Get fields to use for this collection
    fields = COLLECTIONS.get(collection_name, [])
    if not fields:
        print(f"❌ Unknown collection: {collection_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing collection: {collection_name}")
    print(f"Fields to index: {', '.join(fields)}")
    print(f"{'='*60}\n")
    
    # Find ALL documents (to update existing embeddings with all fields)
    # Change to {"embedding": {"$exists": False}} if you only want to process new docs
    query = {}  # Process all documents
    total_docs = collection.count_documents(query)
    
    if total_docs == 0:
        print(f"✅ No documents found in '{collection_name}'!")
        return
    
    print(f"Found {total_docs} documents to process (will update existing embeddings)\n")
    
    # Process in batches
    cursor = collection.find(query)
    processed = 0
    updated = 0
    
    batch_docs = []
    batch_texts = []
    
    for doc in cursor:
        # Create searchable text
        searchable_text = create_searchable_text(doc, fields, collection_name)
        
        if not searchable_text.strip():
            print(f"⚠️  Skipping document {doc.get('_id')}: no text content")
            continue
        
        batch_docs.append(doc)
        batch_texts.append(searchable_text)
        
        # Process batch when full
        if len(batch_docs) >= batch_size:
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            
            # Update documents
            for i, doc in enumerate(batch_docs):
                if not dry_run:
                    collection.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "searchable_text": batch_texts[i],
                                "embedding": embeddings[i].tolist()
                            }
                        }
                    )
                    updated += 1
                else:
                    print(f"[DRY RUN] Would update: {batch_texts[i][:80]}...")
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed: {processed}/{total_docs} ({processed*100//total_docs}%)")
            
            batch_docs = []
            batch_texts = []
    
    # Process remaining documents
    if batch_docs:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        for i, doc in enumerate(batch_docs):
            if not dry_run:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "searchable_text": batch_texts[i],
                            "embedding": embeddings[i].tolist()
                        }
                    }
                )
                updated += 1
            else:
                print(f"[DRY RUN] Would update: {batch_texts[i][:80]}...")
            
            processed += 1
    
    print(f"\n✅ Completed: {updated} documents updated")
    if dry_run:
        print("   (This was a dry run - no actual changes made)")


def main():
    parser = argparse.ArgumentParser(
        description="Add embeddings to MongoDB documents for semantic search"
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=list(COLLECTIONS.keys()),
        help="Specific collection to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all collections"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without updating database"
    )
    
    args = parser.parse_args()
    
    if not args.collection and not args.all:
        parser.error("Must specify either --collection or --all")
    
    # Load embedding model
    print(f"\n📦 Loading embedding model: {args.model}")
    print("   (First run will download ~90MB model)")
    try:
        model = SentenceTransformer(args.model)
        print(f"✅ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Process collections
    if args.all:
        collections_to_process = list(COLLECTIONS.keys())
    else:
        collections_to_process = [args.collection]
    
    for collection_name in collections_to_process:
        try:
            add_embeddings_to_collection(
                collection_name,
                model,
                batch_size=args.batch_size,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"❌ Error processing {collection_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("🎉 Migration complete!")
    print(f"{'='*60}\n")
    
    if not args.dry_run:
        print("Next steps:")
        print("1. Verify embeddings: Check a few documents in MongoDB")
        print("2. Create vector index (if using MongoDB Atlas)")
        print("3. Implement semantic_retrieval() function in main.py")


if __name__ == "__main__":
    main()
