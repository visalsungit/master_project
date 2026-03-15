"""
Script to connect to MongoDB, inspect collections, and generate test data for test_queries.json
Generates 25 test queries for each of the 4 collections: loan, credit_card, fixed_deposit, saving_accounts
"""

import os
import json
from pymongo import MongoClient
from typing import List, Dict, Any
import random

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "banking_db"


def connect_to_db():
    """Connect to MongoDB and return database instance"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        print(f"✅ Connected to MongoDB: {DB_NAME}")
        return db, client
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None, None


def inspect_collection(db, collection_name: str) -> Dict[str, Any]:
    """Inspect a collection and return schema information"""
    try:
        collection = db[collection_name]
        
        # Get collection stats
        count = collection.count_documents({})
        print(f"\n📊 Collection: {collection_name}")
        print(f"   Total documents: {count}")
        
        if count == 0:
            print(f"   ⚠️  Collection is empty!")
            return {"name": collection_name, "count": 0, "fields": [], "sample_docs": []}
        
        # Get sample document to inspect fields
        sample_doc = collection.find_one()
        fields = list(sample_doc.keys()) if sample_doc else []
        
        print(f"   Fields: {', '.join(fields)}")
        
        # Get multiple sample documents for test data generation
        sample_docs = list(collection.find().limit(10))
        
        return {
            "name": collection_name,
            "count": count,
            "fields": fields,
            "sample_docs": sample_docs
        }
    except Exception as e:
        print(f"   ❌ Error inspecting collection: {e}")
        return {"name": collection_name, "count": 0, "fields": [], "sample_docs": []}


def generate_loan_queries(sample_docs: List[Dict]) -> List[Dict]:
    """Generate 25 test queries for loan collection"""
    queries = []
    
    # Query templates for loans
    loan_query_templates = [
        ("personal loan with low interest rate", "looking for competitive personal loan rates"),
        ("home loan for property purchase", "mortgage loan for buying house"),
        ("business loan for SME expansion", "commercial loan for small business"),
        ("car loan with flexible terms", "auto financing with good rates"),
        ("education loan for university", "student loan for higher education"),
        ("loan with quick approval process", "fast loan approval needed"),
        ("secured loan with collateral", "loan requiring property collateral"),
        ("unsecured personal loan", "loan without collateral requirement"),
        ("refinancing existing loan", "loan refinance with better terms"),
        ("short term loan 1 year", "quick loan with short repayment"),
        ("long term loan 10 years", "loan with extended repayment period"),
        ("loan for construction project", "construction financing needed"),
        ("working capital loan", "business working capital financing"),
        ("equipment financing loan", "loan for purchasing business equipment"),
        ("loan with low processing fees", "minimal fees loan product"),
        ("high LTV ratio loan", "loan with high loan-to-value ratio"),
        ("loan for government employees", "special loan for civil servants"),
        ("agricultural loan for farming", "farming loan for crop production"),
        ("microfinance loan small amount", "small business microfinance"),
        ("loan with flexible repayment", "adjustable payment schedule loan"),
        ("USD currency loan", "loan in US dollars"),
        ("KHR currency loan", "loan in Cambodian Riel"),
        ("loan with insurance included", "loan package with insurance coverage"),
        ("loan without insurance requirement", "loan with no mandatory insurance"),
        ("competitive interest rate loan", "best interest rates comparison"),
    ]
    
    for i, (query, alt_query) in enumerate(loan_query_templates):
        # Try to match query with actual documents
        relevant_docs = []
        if sample_docs:
            for doc in sample_docs[:3]:
                doc_id = f"{doc.get('bank_code', 'BANK')}_{doc.get('loan_name', 'Product')}"
                relevant_docs.append(doc_id)
        
        queries.append({
            "query": query if i % 2 == 0 else alt_query,
            "collection": "loan",
            "relevant_docs": relevant_docs[:3] if relevant_docs else ["Sample_Loan_Product"],
            "notes": "Auto-generated test query for loan products"
        })
    
    return queries


def generate_credit_card_queries(sample_docs: List[Dict]) -> List[Dict]:
    """Generate 25 test queries for credit_cards collection"""
    queries = []
    
    card_query_templates = [
        ("premium credit card with benefits", "platinum credit card features"),
        ("visa credit card with cashback", "visa card with rewards program"),
        ("mastercard with travel benefits", "mastercard for frequent travelers"),
        ("credit card with no annual fee", "free credit card no yearly charges"),
        ("gold credit card benefits", "gold tier credit card perks"),
        ("platinum credit card exclusive", "premium platinum card advantages"),
        ("credit card for online shopping", "best card for e-commerce purchases"),
        ("credit card with airport lounge access", "card with travel lounge benefits"),
        ("low interest rate credit card", "credit card with competitive APR"),
        ("high credit limit card", "credit card with large spending limit"),
        ("credit card with insurance coverage", "card with purchase protection"),
        ("contactless payment credit card", "tap and pay enabled card"),
        ("credit card rewards points", "points accumulation credit card"),
        ("cashback credit card deals", "cash rebate credit card"),
        ("business credit card features", "corporate credit card for companies"),
        ("student credit card benefits", "credit card for university students"),
        ("credit card with dining privileges", "restaurant discounts credit card"),
        ("credit card fuel discounts", "petrol station rewards card"),
        ("credit card installment plan", "card with 0% installment offers"),
        ("USD credit card account", "credit card in US dollars"),
        ("multi-currency credit card", "card supporting multiple currencies"),
        ("credit card balance transfer", "low rate balance transfer card"),
        ("secured credit card deposit", "credit card requiring security deposit"),
        ("virtual credit card online", "digital credit card for online use"),
        ("credit card approval requirements", "easy approval credit card application"),
    ]
    
    for i, (query, alt_query) in enumerate(card_query_templates):
        relevant_docs = []
        if sample_docs:
            for doc in sample_docs[:3]:
                doc_id = f"{doc.get('bank_code', 'BANK')}_{doc.get('card_name', 'Card')}"
                relevant_docs.append(doc_id)
        
        queries.append({
            "query": query if i % 2 == 0 else alt_query,
            "collection": "credit_cards",
            "relevant_docs": relevant_docs[:3] if relevant_docs else ["Sample_Credit_Card"],
            "notes": "Auto-generated test query for credit card products"
        })
    
    return queries


def generate_fixed_deposit_queries(sample_docs: List[Dict]) -> List[Dict]:
    """Generate 25 test queries for fixed_deposits collection"""
    queries = []
    
    fd_query_templates = [
        ("fixed deposit with high interest rate", "term deposit best interest rates"),
        ("short term fixed deposit 6 months", "6 month term deposit rates"),
        ("long term fixed deposit 5 years", "5 year fixed deposit investment"),
        ("monthly interest payment FD", "fixed deposit with monthly payouts"),
        ("quarterly interest fixed deposit", "FD with quarterly interest distribution"),
        ("fixed deposit in USD currency", "US dollar term deposit"),
        ("fixed deposit in KHR currency", "Riel term deposit account"),
        ("minimum deposit requirement FD", "low minimum fixed deposit"),
        ("high value fixed deposit", "large amount term deposit"),
        ("fixed deposit for seniors", "senior citizen fixed deposit benefits"),
        ("fixed deposit early withdrawal", "FD with flexible withdrawal options"),
        ("auto-renewal fixed deposit", "automatic rollover term deposit"),
        ("fixed deposit with insurance", "insured fixed deposit account"),
        ("promotional fixed deposit rates", "special offer term deposit"),
        ("corporate fixed deposit", "business term deposit account"),
        ("individual fixed deposit", "personal term deposit investment"),
        ("cumulative fixed deposit", "compound interest term deposit"),
        ("non-cumulative fixed deposit", "regular interest payout FD"),
        ("tax saving fixed deposit", "tax benefit term deposit"),
        ("fixed deposit loan facility", "FD with overdraft option"),
        ("online fixed deposit opening", "digital term deposit account"),
        ("fixed deposit maturity options", "FD tenure and maturity choices"),
        ("fixed deposit interest calculation", "FD interest rate comparison"),
        ("fixed deposit vs savings account", "term deposit versus savings"),
        ("best fixed deposit rates comparison", "highest FD interest rates"),
    ]
    
    for i, (query, alt_query) in enumerate(fd_query_templates):
        relevant_docs = []
        if sample_docs:
            for doc in sample_docs[:3]:
                doc_id = f"{doc.get('bank_code', 'BANK')}_{doc.get('product_name', 'FD')}"
                relevant_docs.append(doc_id)
        
        queries.append({
            "query": query if i % 2 == 0 else alt_query,
            "collection": "fixed_deposits",
            "relevant_docs": relevant_docs[:3] if relevant_docs else ["Sample_Fixed_Deposit"],
            "notes": "Auto-generated test query for fixed deposit products"
        })
    
    return queries


def generate_savings_account_queries(sample_docs: List[Dict]) -> List[Dict]:
    """Generate 25 test queries for savings_accounts collection"""
    queries = []
    
    savings_query_templates = [
        ("savings account with high interest", "best savings account interest rates"),
        ("current account for business", "business current account features"),
        ("savings account no minimum balance", "zero balance savings account"),
        ("CASA account with debit card", "savings account with ATM card"),
        ("online savings account opening", "digital savings account application"),
        ("savings account for children", "kids savings account benefits"),
        ("salary account benefits", "salary crediting savings account"),
        ("premium savings account", "high value savings account perks"),
        ("basic savings account", "simple savings account features"),
        ("savings account with mobile banking", "savings with mobile app access"),
        ("savings account USD currency", "US dollar savings account"),
        ("savings account KHR currency", "Riel savings account"),
        ("multi-currency savings account", "savings in multiple currencies"),
        ("savings account with free transfers", "no fee money transfer savings"),
        ("savings account with checkbook", "savings account check facility"),
        ("savings account overdraft facility", "savings with overdraft protection"),
        ("savings account with insurance", "insured savings account deposit"),
        ("savings account monthly statements", "savings with regular statements"),
        ("savings account for students", "student savings account benefits"),
        ("savings account for seniors", "senior citizen savings benefits"),
        ("joint savings account", "shared savings account opening"),
        ("individual savings account", "single holder savings account"),
        ("savings account interest payout", "savings interest calculation"),
        ("savings account withdrawal limits", "savings transaction limits"),
        ("best savings account comparison", "compare savings account features"),
    ]
    
    for i, (query, alt_query) in enumerate(savings_query_templates):
        relevant_docs = []
        if sample_docs:
            for doc in sample_docs[:3]:
                doc_id = f"{doc.get('bank_code', 'BANK')}_{doc.get('product_name', 'Savings')}"
                relevant_docs.append(doc_id)
        
        queries.append({
            "query": query if i % 2 == 0 else alt_query,
            "collection": "savings_accounts",
            "relevant_docs": relevant_docs[:3] if relevant_docs else ["Sample_Savings_Account"],
            "notes": "Auto-generated test query for savings account products"
        })
    
    return queries


def main():
    """Main function to inspect collections and generate test data"""
    
    # Connect to database
    db, client = connect_to_db()
    if db is None:
        return
    
    try:
        # Define collections to inspect
        collections_map = {
            "loan": "loan",
            "credit_cards": "credit_card",
            "fixed_deposits": "fixed_deposit",
            "savings_accounts": "saving_accounts"
        }
        
        print("\n" + "="*60)
        print("INSPECTING MONGODB COLLECTIONS")
        print("="*60)
        
        # Inspect all collections
        collection_data = {}
        for db_collection, test_name in collections_map.items():
            data = inspect_collection(db, db_collection)
            collection_data[test_name] = data
        
        print("\n" + "="*60)
        print("GENERATING TEST QUERIES")
        print("="*60)
        
        # Generate test queries for each collection
        all_queries = []
        
        # Loan queries (25)
        loan_data = collection_data.get("loan", {})
        loan_queries = generate_loan_queries(loan_data.get("sample_docs", []))
        all_queries.extend(loan_queries)
        print(f"✅ Generated {len(loan_queries)} loan queries")
        
        # Credit card queries (25)
        card_data = collection_data.get("credit_card", {})
        card_queries = generate_credit_card_queries(card_data.get("sample_docs", []))
        all_queries.extend(card_queries)
        print(f"✅ Generated {len(card_queries)} credit card queries")
        
        # Fixed deposit queries (25)
        fd_data = collection_data.get("fixed_deposit", {})
        fd_queries = generate_fixed_deposit_queries(fd_data.get("sample_docs", []))
        all_queries.extend(fd_queries)
        print(f"✅ Generated {len(fd_queries)} fixed deposit queries")
        
        # Savings account queries (25)
        savings_data = collection_data.get("saving_accounts", {})
        savings_queries = generate_savings_account_queries(savings_data.get("sample_docs", []))
        all_queries.extend(savings_queries)
        print(f"✅ Generated {len(savings_queries)} savings account queries")
        
        print(f"\n📊 Total queries generated: {len(all_queries)}")
        
        # Save to test_queries.json
        output_path = "/Users/visalsun/dev/ollama_chat/blocks/experiments/test_queries.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_queries, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Test queries saved to: {output_path}")
        print(f"   Total: {len(all_queries)} queries")
        print(f"   - Loan: 25 queries")
        print(f"   - Credit Cards: 25 queries")
        print(f"   - Fixed Deposits: 25 queries")
        print(f"   - Savings Accounts: 25 queries")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.close()
            print("\n🔌 Disconnected from MongoDB")


if __name__ == "__main__":
    main()
