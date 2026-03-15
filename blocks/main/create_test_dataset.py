"""
Test Dataset Generator
======================

Helper script to create and manage test datasets for evaluation.
This will help you build a comprehensive test set for your thesis.

Usage:
    python create_test_dataset.py --interactive
    python create_test_dataset.py --auto-generate --output test_queries.json
"""

import json
import os
from typing import List, Dict, Any
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "banking_db"


def get_all_products_by_collection() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all products from each collection for reference."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    collections = {
        "loan": list(db.loan.find({}, {"_id": 0, "bank": 1, "loan_name": 1, "bank_code": 1})),
        "credit_cards": list(db.credit_cards.find({}, {"_id": 0, "bank": 1, "card_name": 1, "bank_code": 1})),
        "fixed_deposits": list(db.fixed_deposits.find({}, {"_id": 0, "bank_name": 1, "product_name": 1, "bank_code": 1})),
        "savings_accounts": list(db.savings_accounts.find({}, {"_id": 0, "bank_name": 1, "product_name": 1, "bank_code": 1})),
        "exchange_rates": list(db.exchange_rates.find({}, {"_id": 0, "bank_name": 1, "bank_code": 1, "rate_type": 1})),
    }
    
    client.close()
    return collections


def doc_id(doc: Dict[str, Any], collection: str) -> str:
    """Generate document ID."""
    if collection == "loan":
        return f"{doc.get('bank_code', 'UNKNOWN')}_{doc.get('loan_name', 'UNKNOWN')}"
    elif collection == "credit_cards":
        return f"{doc.get('bank_code', 'UNKNOWN')}_{doc.get('card_name', 'UNKNOWN')}"
    elif collection in ["fixed_deposits", "savings_accounts", "exchange_rates"]:
        name = doc.get('product_name', doc.get('rate_type', 'UNKNOWN'))
        return f"{doc.get('bank_code', 'UNKNOWN')}_{name}"
    return "UNKNOWN"


def auto_generate_test_queries() -> List[Dict[str, Any]]:
    """
    Auto-generate basic test queries from existing data.
    These should be manually reviewed and expanded.
    """
    print("Auto-generating test queries from database...")
    
    products = get_all_products_by_collection()
    test_queries = []
    
    # Loan queries
    print("\n1. Generating loan queries...")
    loan_templates = [
        ("personal loan with low interest", "personal", ["USD"]),
        ("home loan for property purchase", "home", ["USD"]),
        ("business loan for SME", "business", ["USD", "KHR"]),
        ("unsecured loan quick approval", "unsecured", ["USD"]),
        ("mortgage loan competitive rate", "mortgage", ["USD"]),
    ]
    
    for query_text, keyword, currencies in loan_templates:
        # Find relevant loans
        relevant = []
        for loan in products["loan"]:
            loan_name = str(loan.get("loan_name", "")).lower()
            if keyword in loan_name:
                relevant.append(doc_id(loan, "loan"))
        
        if relevant:
            test_queries.append({
                "query": query_text,
                "collection": "loan",
                "relevant_docs": relevant[:3],  # Top 3
                "currency": currencies[0] if currencies else None,
                "notes": f"Auto-generated: {keyword} keyword match"
            })
    
    # Credit card queries
    print("2. Generating credit card queries...")
    card_templates = [
        ("credit card no annual fee", ["free", "no fee"]),
        ("premium credit card with benefits", ["premium", "platinum", "gold"]),
        ("visa credit card", ["visa"]),
        ("mastercard with cashback", ["mastercard"]),
    ]
    
    for query_text, keywords in card_templates:
        relevant = []
        for card in products["credit_cards"]:
            card_name = str(card.get("card_name", "")).lower()
            if any(kw in card_name for kw in keywords):
                relevant.append(doc_id(card, "credit_cards"))
        
        if relevant:
            test_queries.append({
                "query": query_text,
                "collection": "credit_cards",
                "relevant_docs": relevant[:3],
                "notes": f"Auto-generated: keyword match"
            })
    
    # Fixed deposit queries
    print("3. Generating fixed deposit queries...")
    fd_templates = [
        "fixed deposit with monthly interest payment",
        "term deposit high interest rate",
        "fixed deposit USD currency",
    ]
    
    for query_text in fd_templates:
        # For FD, just list all as potentially relevant
        relevant = [doc_id(fd, "fixed_deposits") for fd in products["fixed_deposits"][:3]]
        test_queries.append({
            "query": query_text,
            "collection": "fixed_deposits",
            "relevant_docs": relevant,
            "notes": "Auto-generated: needs manual verification"
        })
    
    # Exchange rate queries
    print("4. Generating exchange rate queries...")
    fx_templates = [
        "best USD to KHR exchange rate",
        "current exchange rate for USD",
        "forex rate comparison",
    ]
    
    for query_text in fx_templates:
        relevant = [doc_id(fx, "exchange_rates") for fx in products["exchange_rates"][:3]]
        test_queries.append({
            "query": query_text,
            "collection": "exchange_rates",
            "relevant_docs": relevant,
            "notes": "Auto-generated: needs manual verification"
        })
    
    # Savings account queries
    print("5. Generating savings account queries...")
    savings_templates = [
        "savings account with high interest",
        "current account for business",
        "CASA account with debit card",
    ]
    
    for query_text in savings_templates:
        relevant = [doc_id(sa, "savings_accounts") for sa in products["savings_accounts"][:2]]
        test_queries.append({
            "query": query_text,
            "collection": "savings_accounts",
            "relevant_docs": relevant,
            "notes": "Auto-generated: needs manual verification"
        })
    
    print(f"\n✅ Generated {len(test_queries)} test queries")
    return test_queries


def interactive_test_builder():
    """Interactive mode to build test dataset."""
    print("\n" + "="*60)
    print("INTERACTIVE TEST DATASET BUILDER")
    print("="*60)
    
    products = get_all_products_by_collection()
    test_queries = []
    
    print("\nAvailable collections:")
    for i, coll in enumerate(products.keys(), 1):
        print(f"{i}. {coll} ({len(products[coll])} products)")
    
    while True:
        print("\n" + "-"*60)
        print(f"Current test set: {len(test_queries)} queries")
        
        action = input("\n[A]dd query, [L]ist products, [S]ave, [Q]uit: ").strip().upper()
        
        if action == "Q":
            break
        
        elif action == "A":
            # Add new test query
            query = input("Enter query: ").strip()
            if not query:
                continue
            
            coll = input("Collection (loan/credit_cards/fixed_deposits/savings_accounts/exchange_rates): ").strip()
            if coll not in products:
                print(f"❌ Invalid collection: {coll}")
                continue
            
            print(f"\nAvailable products in {coll}:")
            for i, prod in enumerate(products[coll][:10], 1):
                print(f"{i}. {doc_id(prod, coll)}")
            
            relevant_input = input("\nRelevant doc IDs (comma-separated, or numbers): ").strip()
            relevant_docs = []
            
            if relevant_input:
                for item in relevant_input.split(","):
                    item = item.strip()
                    if item.isdigit():
                        idx = int(item) - 1
                        if 0 <= idx < len(products[coll]):
                            relevant_docs.append(doc_id(products[coll][idx], coll))
                    else:
                        relevant_docs.append(item)
            
            currency = input("Currency filter (optional, e.g., USD): ").strip().upper() or None
            
            test_queries.append({
                "query": query,
                "collection": coll,
                "relevant_docs": relevant_docs,
                "currency": currency
            })
            
            print(f"✅ Added query #{len(test_queries)}")
        
        elif action == "L":
            # List products
            coll = input("Collection name: ").strip()
            if coll in products:
                print(f"\nProducts in {coll}:")
                for i, prod in enumerate(products[coll], 1):
                    print(f"{i}. {doc_id(prod, coll)}")
            else:
                print(f"❌ Unknown collection: {coll}")
        
        elif action == "S":
            # Save
            filename = input("Output filename (default: test_queries.json): ").strip() or "test_queries.json"
            with open(filename, "w") as f:
                json.dump(test_queries, f, indent=2)
            print(f"✅ Saved {len(test_queries)} queries to {filename}")
    
    return test_queries


def save_test_dataset(queries: List[Dict[str, Any]], filename: str):
    """Save test dataset to JSON file."""
    with open(filename, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"\n✅ Saved {len(queries)} test queries to {filename}")


def load_test_dataset(filename: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test dataset for retrieval evaluation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--auto-generate", action="store_true", help="Auto-generate queries")
    parser.add_argument("--output", default="test_queries.json", help="Output filename")
    
    args = parser.parse_args()
    
    if args.interactive:
        queries = interactive_test_builder()
        if queries:
            save_test_dataset(queries, args.output)
    
    elif args.auto_generate:
        queries = auto_generate_test_queries()
        save_test_dataset(queries, args.output)
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print(f"1. Review and edit {args.output}")
        print("2. Add more diverse queries (aim for 50-100)")
        print("3. Manually verify relevant documents")
        print("4. Run evaluation:")
        print(f"   python evaluation_framework.py --dataset {args.output}")
    
    else:
        parser.print_help()
