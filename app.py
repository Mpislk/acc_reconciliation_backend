# backend_app.py
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import traceback
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from datetime import timedelta


# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configuration ---
UPLOAD_FOLDER = '../uploads' # Relative path to a folder outside the app root
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Matching Parameters (Adjust as needed) ---
AMOUNT_TOLERANCE = 500 # Allow for very small discrepancies in amount (e.g., due to rounding)
DATE_TOLERANCE_DAYS = 5 # Allow dates to be off by up to 5 days
SEMANTIC_SIMILARITY_THRESHOLD = 0.75 # Cosine similarity threshold for semantic match (0.0 to 1.0)

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_file(filepath, file_type):
    """Parses uploaded CSV/Excel files into a list of dictionaries."""
    try:
        print(f"\n--- Parsing {file_type} ---")
        print(f"File path: {filepath}")

        # Read file based on extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else: # Excel files
            df = pd.read_excel(filepath, sheet_name=0, header=0) # Read first sheet

        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")

        # Convert to dictionary format for JSON serialization
        # Replace NaN with empty string for JSON compatibility before converting to dict
        data = df.fillna("").to_dict('records')

        return {
            'success': True,
            'data': data,
            'columns': list(df.columns),
            'row_count': len(df),
            'file_type': file_type
        }
    except Exception as e:
        print(f"Error parsing {file_type}: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'file_type': file_type
        }

def preprocess_data(bank_data_dict, invoice_data_dict):
    """
    Cleans and standardizes data from both bank and invoice files for matching.
    Adds 'Description_Clean', 'Amount', 'Date' in standardized formats.
    """
    print("\n--- Starting Data Preprocessing ---")

    # Ensure data is not empty before creating DataFrame
    bank_df = pd.DataFrame(bank_data_dict['data']) if bank_data_dict and bank_data_dict['data'] else pd.DataFrame()
    invoice_df = pd.DataFrame(invoice_data_dict['data']) if invoice_data_dict and invoice_data_dict['data'] else pd.DataFrame()

    if bank_df.empty or invoice_df.empty:
        print("Warning: One or both DataFrames are empty after initial loading.")
        return {
            'bank_data_processed': [],
            'invoice_data_processed': [],
            'preprocessing_info': {
                'bank_original_rows': len(bank_df),
                'bank_processed_rows': 0,
                'invoice_original_rows': len(invoice_df),
                'invoice_processed_rows': 0
            }
        }

    # --- BANK STATEMENT PREPROCESSING ---
    # Remove columns that are completely empty (as per requirement: Bank ID, Account Number, End to End ID)
    bank_df_initial_shape = bank_df.shape
    bank_df = bank_df.dropna(axis=1, how='all')
    print(f"Bank statement shape after dropping all-NaN columns: {bank_df_initial_shape} -> {bank_df.shape}")

    # Standardize 'Date' column
    if 'Date' in bank_df.columns:
        bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
    else:
        bank_df['Date'] = pd.NaT # Set to Not a Time if column missing

    # Combine Credit and Debit amounts into a single 'Amount' column
    if 'Credit Amount' in bank_df.columns and 'Debit Amount' in bank_df.columns:
        bank_df['Credit Amount'] = pd.to_numeric(bank_df['Credit Amount'], errors='coerce').fillna(0)
        bank_df['Debit Amount'] = pd.to_numeric(bank_df['Debit Amount'], errors='coerce').fillna(0)
        bank_df['Amount'] = bank_df['Credit Amount'] - bank_df['Debit Amount'] # Credit as positive, Debit as negative
    elif 'Amount' in bank_df.columns: # If there's already a single 'Amount' column
        bank_df['Amount'] = pd.to_numeric(bank_df['Amount'], errors='coerce').fillna(0)
    else:
        bank_df['Amount'] = 0.0 # Default if no amount column

    # --- ENHANCEMENT: Combine relevant text fields for Bank Statement Description_Clean ---
    # This ensures names, KAI numbers, and other details from various columns are included for embedding
    bank_df['Description_Clean'] = bank_df.apply(lambda row:
        " ".join(filter(None, [
            str(row.get('Description', '')).strip(), # Primary description, often contains names & KAI
            str(row.get('Tran Type', '')).strip(),
            str(row.get('Customer Ref #', '')).strip(), # Might contain hidden dates or KAI-like numbers
            str(row.get('Bank Ref #', '')).strip(),
            str(row.get('Account Title', '')).strip(),
            str(row.get('Account Owner', '')).strip()
        ])).lower(), axis=1
    )
    bank_df['Description_Clean'] = bank_df['Description_Clean'].str.replace(r'\s+', ' ', regex=True).str.strip()


    # --- INVOICE PREPROCESSING ---
    invoice_df_initial_shape = invoice_df.shape
    invoice_df = invoice_df.dropna(axis=1, how='all')
    print(f"Invoices shape after dropping all-NaN columns: {invoice_df_initial_shape} -> {invoice_df.shape}")

    # Standardize 'Date' column
    if 'Date' in invoice_df.columns:
        invoice_df['Date'] = pd.to_datetime(invoice_df['Date'], errors='coerce')
    else:
        invoice_df['Date'] = pd.NaT

    # Ensure 'Amount' is numeric
    if 'Amount' in invoice_df.columns:
        invoice_df['Amount'] = pd.to_numeric(invoice_df['Amount'], errors='coerce').fillna(0)
    else:
        invoice_df['Amount'] = 0.0

    # --- ENHANCEMENT: Combine relevant text fields for Invoice Description_Clean ---
    # This ensures Customer names and KAI numbers (from No.) are included for embedding
    invoice_df['Description_Clean'] = invoice_df.apply(lambda row:
        " ".join(filter(None, [
            str(row.get('Type', '')).strip(),
            str(row.get('No.', '')).strip(), # Contains the 'KAI' number for invoices
            str(row.get('Customer', '')).strip(), # Contains the customer name
            str(row.get('Memo', '')).strip() # Include 'Memo' column content if available
        ])).lower(),
        axis=1
    )
    invoice_df['Description_Clean'] = invoice_df['Description_Clean'].str.replace(r'\s+', ' ', regex=True).str.strip()


    # Remove rows with invalid dates or amounts after conversion
    # And filter out entries where cleaned description is empty or just whitespace
    bank_df_cleaned = bank_df.dropna(subset=['Date', 'Amount']).copy()
    bank_df_cleaned = bank_df_cleaned[bank_df_cleaned['Description_Clean'].str.strip() != ''].copy()

    invoice_df_cleaned = invoice_df.dropna(subset=['Date', 'Amount']).copy()
    invoice_df_cleaned = invoice_df_cleaned[invoice_df_cleaned['Description_Clean'].str.strip() != ''].copy()


    print(f"Bank statement after preprocessing: {bank_df_cleaned.shape}")
    print(f"Invoices after preprocessing: {invoice_df_cleaned.shape}")

    return {
        'bank_data_processed': bank_df_cleaned.to_dict('records'),
        'invoice_data_processed': invoice_df_cleaned.to_dict('records'),
        'preprocessing_info': {
            'bank_original_rows': len(bank_df),
            'bank_processed_rows': len(bank_df_cleaned),
            'invoice_original_rows': len(invoice_df),
            'invoice_processed_rows': len(invoice_df_cleaned)
        }
    }

def get_embedding(text, model="models/embedding-001"):
    """Generates an embedding for the given text using the specified Gemini model."""
    if not text or not isinstance(text, str) or text.strip() == "":
        return None # Handle empty, non-string, or whitespace-only text gracefully
    try:
        # Gemini embedding models have a rate limit, handle potential errors
        result = genai.embed_content(model=model, content=text)
        return result["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        return None

def calculate_confidence(amount_matched, date_matched, semantic_score):
    """
    Calculates a confidence score for a match based on criteria,
    prioritizing amount and semantic similarity (name/KAI) over date.
    """
    score = 0.0
    reason_parts = []

    # Amount match is a very strong indicator
    if amount_matched:
        score += 0.45 # High weight
        reason_parts.append("Amount matched")

    # Semantic similarity is critical for name/KAI number matching and is given high importance
    if semantic_score is not None and semantic_score >= SEMANTIC_SIMILARITY_THRESHOLD:
        score += 0.45 * semantic_score # High weight, scaled by the actual similarity score
        reason_parts.append(f"Strong semantic match (score: {semantic_score:.2f})")
    elif semantic_score is not None:
         reason_parts.append(f"Semantic similarity low (score: {semantic_score:.2f})")

    # Date proximity is now given lower importance as per user's guidance
    if date_matched:
        score += 0.10 # Reduced weight
        reason_parts.append("Date matched (within tolerance)")

    # Normalize score to 0-1 range or cap at 1.0
    confidence = min(score, 1.0)
    reason = ", ".join(reason_parts) if reason_parts else "No strong matching criteria met"
    return confidence, reason

# --- Flask Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "Backend is running and healthy!"}), 200

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads, saves them, parses, and preprocesses their content."""
    try:
        if 'bank_statement' not in request.files or 'invoices' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both bank_statement and invoices files are required.'
            }), 400

        bank_file = request.files['bank_statement']
        invoice_file = request.files['invoices']

        if bank_file.filename == '' or invoice_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'One or both file selections are empty. Please select valid files.'
            }), 400

        if not (allowed_file(bank_file.filename) and allowed_file(invoice_file.filename)):
            return jsonify({
                'success': False,
                'error': 'Unsupported file type. Only CSV and Excel (xlsx, xls) files are allowed.'
            }), 400

        bank_filename = secure_filename(bank_file.filename)
        invoice_filename = secure_filename(invoice_file.filename)

        bank_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"bank_{bank_filename}")
        invoice_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"invoice_{invoice_filename}")

        bank_file.save(bank_filepath)
        invoice_file.save(invoice_filepath)
        print(f"Saved bank file to: {bank_filepath}")
        print(f"Saved invoice file to: {invoice_filepath}")

        bank_data_parsed_result = parse_file(bank_filepath, 'bank_statement')
        invoice_data_parsed_result = parse_file(invoice_filepath, 'invoices')

        if not bank_data_parsed_result['success'] or not invoice_data_parsed_result['success']:
            return jsonify({
                'success': False,
                'error': f"Error parsing files: Bank: {bank_data_parsed_result.get('error', 'N/A')}, Invoice: {invoice_data_parsed_result.get('error', 'N/A')}"
            }), 400

        # Perform preprocessing immediately after successful parsing
        preprocessed_data = preprocess_data(bank_data_parsed_result, invoice_data_parsed_result)

        response_data = {
            'success': True,
            'message': 'Files uploaded and preprocessed successfully',
            'bank_statement_original': { # Keep original parsed data for reference if needed
                'filename': bank_filename,
                'columns': bank_data_parsed_result['columns'],
                'row_count': bank_data_parsed_result['row_count'],
                'data': bank_data_parsed_result['data']
            },
            'invoices_original': { # Keep original parsed data for reference if needed
                'filename': invoice_filename,
                'columns': invoice_data_parsed_result['columns'],
                'row_count': invoice_data_parsed_result['row_count'],
                'data': invoice_data_parsed_result['data']
            },
            'bank_statement_processed': preprocessed_data['bank_data_processed'],
            'invoices_processed': preprocessed_data['invoice_data_processed'],
            'preprocessing_info': preprocessed_data['preprocessing_info']
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error in /upload endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error during file upload: {str(e)}'
        }), 500


@app.route('/match', methods=['POST'])
def match_transactions():
    """
    API endpoint to perform AI-powered matching between bank transactions and invoices.
    Expects preprocessed bank and invoice data.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'success': False, 'error': 'No JSON data received for matching.'}), 400

        bank_data_raw = request_data.get('bank_data')
        invoice_data_raw = request_data.get('invoice_data')

        if not bank_data_raw or not invoice_data_raw:
            return jsonify({'success': False, 'error': 'Missing bank_data or invoice_data in request.'}), 400

        # Convert to DataFrame for easier processing
        bank_df = pd.DataFrame(bank_data_raw)
        invoice_df = pd.DataFrame(invoice_data_raw)

        # --- CRITICAL FIX: Re-ensure correct data types after JSON roundtrip ---
        # Convert Amount columns to numeric
        bank_df['Amount'] = pd.to_numeric(bank_df['Amount'], errors='coerce')
        invoice_df['Amount'] = pd.to_numeric(invoice_df['Amount'], errors='coerce')

        # Convert Date columns to datetime objects
        bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
        invoice_df['Date'] = pd.to_datetime(invoice_df['Date'], errors='coerce')

        # Filter out rows where crucial columns became NaN/NaT after conversion
        # This prevents type errors later in calculations
        bank_df_filtered = bank_df.dropna(subset=['Amount', 'Date', 'Description_Clean']).reset_index(drop=True)
        invoice_df_filtered = invoice_df.dropna(subset=['Amount', 'Date', 'Description_Clean']).reset_index(drop=True)

        print(f"Bank data types after re-conversion for matching:\n{bank_df_filtered[['Date', 'Amount', 'Description_Clean']].dtypes}")
        print(f"Invoice data types after re-conversion for matching:\n{invoice_df_filtered[['Date', 'Amount', 'Description_Clean']].dtypes}")
        print(f"Bank entries after filtering for matching: {len(bank_df_filtered)}")
        print(f"Invoice entries after filtering for matching: {len(invoice_df_filtered)}")


        # Generate embeddings for descriptions
        bank_df_filtered['embedding'] = bank_df_filtered['Description_Clean'].apply(
            lambda x: get_embedding(x, model="models/embedding-001")
        )
        invoice_df_filtered['embedding'] = invoice_df_filtered['Description_Clean'].apply(
            lambda x: get_embedding(x, model="models/embedding-001")
        )

        # Drop rows where embedding generation failed (embedding is None)
        bank_df_embedded = bank_df_filtered.dropna(subset=['embedding']).reset_index(drop=True)
        invoice_df_embedded = invoice_df_filtered.dropna(subset=['embedding']).reset_index(drop=True)

        print(f"Bank entries with valid embeddings: {len(bank_df_embedded)}")
        print(f"Invoice entries with valid embeddings: {len(invoice_df_embedded)}")

        matched_transactions = []
        unmatched_bank_indices = set(bank_df_embedded.index)
        unmatched_invoice_indices = set(invoice_df_embedded.index)

        # Iterate through bank transactions to find matches
        for bank_idx, bank_row in bank_df_embedded.iterrows():
            if bank_idx not in unmatched_bank_indices:
                continue # Already matched

            best_match = None
            max_confidence = 0.0

            for invoice_idx, invoice_row in invoice_df_embedded.iterrows():
                if invoice_idx not in unmatched_invoice_indices:
                    continue # Already matched

                # --- 1. Amount Match ---
                amount_matched = abs(bank_row['Amount'] - invoice_row['Amount']) <= AMOUNT_TOLERANCE

                # --- 2. Date Match ---
                date_matched = False
                # Ensure dates are not NaT before calculating difference
                if pd.notna(bank_row['Date']) and pd.notna(invoice_row['Date']):
                    date_difference = abs(bank_row['Date'] - invoice_row['Date']).days
                    date_matched = date_difference <= DATE_TOLERANCE_DAYS

                # --- 3. Semantic Similarity Match ---
                semantic_score = None
                if bank_row['embedding'] is not None and invoice_row['embedding'] is not None:
                    # Reshape for cosine_similarity: (1, n_features)
                    bank_embedding = np.array(bank_row['embedding']).reshape(1, -1)
                    invoice_embedding = np.array(invoice_row['embedding']).reshape(1, -1)
                    semantic_score = cosine_similarity(bank_embedding, invoice_embedding)[0][0]

                # Calculate overall confidence based on adjusted priorities
                confidence, match_reason = calculate_confidence(
                    amount_matched, date_matched, semantic_score
                )

                # Prioritize matches with higher confidence and meeting semantic threshold
                if confidence > max_confidence and (semantic_score is None or semantic_score >= SEMANTIC_SIMILARITY_THRESHOLD):
                    best_match = {
                        "bank_idx": bank_idx,
                        "invoice_idx": invoice_idx,
                        "confidence_score": confidence,
                        "match_reason": match_reason
                    }
                    max_confidence = confidence

            if best_match:
                # Add to matched transactions list
                # Convert datetime objects to string for JSON serialization
                file_a_entry = bank_df_embedded.loc[best_match['bank_idx']].drop('embedding').to_dict()
                file_b_entry = invoice_df_embedded.loc[best_match['invoice_idx']].drop('embedding').to_dict()

                # Format Date objects in dicts for JSON to ISO format
                if 'Date' in file_a_entry and pd.notna(file_a_entry['Date']):
                    file_a_entry['Date'] = file_a_entry['Date'].isoformat()
                if 'Date' in file_b_entry and pd.notna(file_b_entry['Date']):
                    file_b_entry['Date'] = file_b_entry['Date'].isoformat()

                matched_transactions.append({
                    "file_a_entry": file_a_entry,
                    "file_b_entry": file_b_entry,
                    "confidence_score": round(best_match['confidence_score'], 4),
                    "match_reason": best_match['match_reason']
                })

                # Mark indices as matched
                unmatched_bank_indices.discard(best_match['bank_idx'])
                unmatched_invoice_indices.discard(best_match['invoice_idx'])

        # Prepare unmatched entries
        unmatched_bank_entries = []
        for idx in unmatched_bank_indices:
            entry = bank_df_embedded.loc[idx].drop('embedding').to_dict()
            if 'Date' in entry and pd.notna(entry['Date']):
                entry['Date'] = entry['Date'].isoformat()
            unmatched_bank_entries.append(entry)

        unmatched_invoice_entries = []
        for idx in unmatched_invoice_indices:
            entry = invoice_df_embedded.loc[idx].drop('embedding').to_dict()
            if 'Date' in entry and pd.notna(entry['Date']):
                entry['Date'] = entry['Date'].isoformat()
            unmatched_invoice_entries.append(entry)

        return jsonify({
            'success': True,
            'message': f'AI Matching completed. Found {len(matched_transactions)} matches.',
            'matches': matched_transactions,
            'unmatched_file_a_entries': unmatched_bank_entries,
            'unmatched_file_b_entries': unmatched_invoice_entries
        }), 200

    except Exception as e:
        print(f"Error in /match endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error during matching: {str(e)}'
        }), 500

if __name__ == '__main__':
    import os
    import logging
    
    # Configure logger for local development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Flask application")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    
    # Get port from environment (for production) or use 5000 (for development)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)