# Database Schema Inspection Report

## MongoDB Connection Details
- **URI**: mongodb://localhost:27017
- **Database**: banking_db
- **Date**: January 4, 2026

## Collection Schemas

### 1. Loan Collection
- **Total Documents**: 74
- **Fields**:
  - _id
  - bank
  - product_type
  - loan_name
  - loan_category
  - description
  - currency
  - loan_amount
  - loan_term
  - interest_rate
  - repayment_modes
  - fees
  - ltv_ratio
  - insurance_requirements
  - acceptable_collateral
  - borrower_eligibility
  - documents_required
  - advantages
  - additional_information
  - source_url
  - updated_date
  - bank_code
  - schema_version
  - embedding
  - searchable_text

### 2. Credit Cards Collection
- **Total Documents**: 15
- **Fields**:
  - _id
  - bank
  - product_type
  - card_name
  - network
  - tier
  - currency
  - fees
  - limits
  - interest
  - expiry_years
  - source_url
  - updated_at
  - bank_code
  - schema_version
  - embedding
  - searchable_text

### 3. Fixed Deposits Collection
- **Total Documents**: 10
- **Fields**:
  - _id
  - bank_name
  - product_name
  - product_type
  - segment
  - currencies_supported
  - interest_payment_options
  - minimum_deposit
  - benefits
  - requirements
  - term_deposit_types
  - notes
  - source_url
  - inserted_at
  - bank_code
  - schema_version
  - embedding
  - searchable_text

### 4. Savings Accounts Collection
- **Total Documents**: 10
- **Fields**:
  - _id
  - product_name
  - bank_name
  - cutoff_date
  - description
  - interest_rate
  - product_type
  - schema_version
  - scraped_at
  - source_url
  - updated_at
  - bank_code
  - embedding
  - searchable_text

## Test Data Generation Summary

### Generated Test Queries: 100 total
- **Loan queries**: 25
- **Credit Card queries**: 25
- **Fixed Deposit queries**: 25
- **Savings Account queries**: 25

### Output File
- **Location**: `/Users/visalsun/dev/ollama_chat/blocks/experiments/test_queries.json`
- **Format**: JSON array with query objects

### Query Object Structure
Each query object contains:
- `query`: The test query string
- `collection`: The target collection name
- `relevant_docs`: Array of relevant document identifiers
- `notes`: Description/metadata about the query

### Sample Queries by Collection

#### Loan Queries (examples)
- "personal loan with low interest rate"
- "home loan for property purchase"
- "business loan for SME expansion"
- "car loan with flexible terms"
- "education loan for university"

#### Credit Card Queries (examples)
- "premium credit card with benefits"
- "visa credit card with cashback"
- "mastercard with travel benefits"
- "credit card with no annual fee"
- "gold credit card benefits"

#### Fixed Deposit Queries (examples)
- "fixed deposit with high interest rate"
- "short term fixed deposit 6 months"
- "long term fixed deposit 5 years"
- "monthly interest payment FD"
- "fixed deposit in USD currency"

#### Savings Account Queries (examples)
- "savings account with high interest"
- "current account for business"
- "savings account no minimum balance"
- "CASA account with debit card"
- "online savings account opening"

## Script Details
- **Script Name**: generate_test_data.py
- **Location**: `/Users/visalsun/dev/ollama_chat/blocks/data_integrations/`
- **Dependencies**: pymongo, json

## Usage
To regenerate test data:
```bash
cd /Users/visalsun/dev/ollama_chat/blocks/data_integrations
python generate_test_data.py
```
