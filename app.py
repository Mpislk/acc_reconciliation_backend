import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import traceback
import json
import logging
from datetime import datetime
import hashlib
import base64
from cryptography.fernet import Fernet
import re
import google.generativeai as genai
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Validate required environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file. Please create one and add your key.")

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
CHATGPT_MODEL = "gpt-3.5-turbo"

# --- Gemini API Configuration ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
# genai.configure(api_key=GEMINI_API_KEY)

# model = genai.GenerativeModel('gemini-1.5-pro')



# Flask app configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Generate encryption key (in production, store this securely)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# --- Helper Functions ---

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    """Read CSV or Excel file and return DataFrame."""
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read CSV file with any supported encoding")
                
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def basic_cleanup(df):
    """Basic cleanup - only remove completely empty rows and clean column names."""
    try:
        # Remove completely empty rows
        df_cleaned = df.dropna(how='all')
        
        # Remove rows where all values are empty strings
        df_cleaned = df_cleaned[~(df_cleaned.astype(str) == '').all(axis=1)]
        
        # Clean column names
        df_cleaned.columns = df_cleaned.columns.str.strip()
        
        logger.info(f"Basic cleanup: {len(df)} -> {len(df_cleaned)} rows")
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error in basic cleanup: {str(e)}")
        raise

def identify_sensitive_columns(df):
    """Identify columns that likely contain sensitive information."""
    sensitive_patterns = [
        r'.*account.*num.*',
        r'.*account.*no.*',
        r'.*acc.*num.*',
        r'.*acc.*no.*',
        r'.*name.*',
        r'.*customer.*',
        r'.*client.*',
        r'.*phone.*',
        r'.*mobile.*',
        r'.*email.*',
        r'.*address.*',
        r'.*ssn.*',
        r'.*social.*security.*',
        r'.*tax.*id.*',
        r'.*ein.*',
        r'.*routing.*',
        r'.*iban.*',
        r'.*swift.*'
    ]
    
    sensitive_columns = []
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in sensitive_patterns:
            if re.match(pattern, col_lower):
                sensitive_columns.append(col)
                break
    
    return sensitive_columns

def encrypt_sensitive_data(df):
    """Encrypt sensitive data in the DataFrame."""
    df_encrypted = df.copy()
    sensitive_columns = identify_sensitive_columns(df)
    
    for col in sensitive_columns:
        if col in df_encrypted.columns:
            df_encrypted[col] = df_encrypted[col].astype(str).apply(
                lambda x: cipher_suite.encrypt(x.encode()).decode() if pd.notna(x) and x != '' else x
            )
    
    logger.info(f"Encrypted sensitive columns: {sensitive_columns}")
    return df_encrypted, sensitive_columns

def preprocess_important_columns(df, important_columns, file_type="general"):
    """Clean and preprocess only the important columns identified by LLM."""
    try:
        df_processed = df.copy()
        
        # Only process the important columns
        for col in important_columns:
            if col not in df_processed.columns:
                continue
                
            col_lower = col.lower()
            
            # Convert numeric columns (amounts, values, etc.)
            if any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'total', 'sum', 'balance', 'debit', 'credit']):
                # Clean numeric data - remove currency symbols, commas, etc.
                df_processed[col] = df_processed[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                logger.info(f"Processed numeric column: {col}")
            
            # Handle date columns
            elif any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'transaction_date', 'posting_date']):
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    logger.info(f"Processed date column: {col}")
                except:
                    logger.warning(f"Could not process date column: {col}")
                    pass
            
            # Handle text columns (descriptions, references)
            elif any(keyword in col_lower for keyword in ['description', 'reference', 'memo', 'notes', 'details']):
                # Clean and standardize text
                df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()
                logger.info(f"Processed text column: {col}")
        
        logger.info(f"Preprocessed important columns for {file_type}: {important_columns}")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error preprocessing important columns for {file_type}: {str(e)}")
        raise

def get_column_identification_prompt(bank_sample, invoice_sample):
    """Create prompt for LLM to identify key columns for matching."""
    prompt = f"""
You are a financial data analyst and expert in account reconciliation. I need you to analyze two datasets and identify the key columns required for transaction matching.

**Bank Statement Sample (first 5 rows):**
{bank_sample.to_string()}

**Invoice Data Sample (first 5 rows):**
{invoice_sample.to_string()}

Please analyze these datasets and identify:
1. Key columns from the bank statement that are essential for transaction matching
2. Key columns from the invoice data that are essential for transaction matching
3. The matching logic that should be used

Return your response in the following JSON format:
{{
    "bank_key_columns": ["column1", "column2", ...],
    "invoice_key_columns": ["column1", "column2", ...],
    "matching_strategy": "Brief explanation of how these transactions should be matched",
    "primary_match_fields": {{
        "bank": "primary_field_name",
        "invoice": "primary_field_name"
    }},
    "secondary_match_fields": {{
        "bank": ["field1", "field2"],
        "invoice": ["field1", "field2"]
    }}
}}

Focus on columns that typically contain:
- Transaction amounts/values
- Dates
- Reference numbers/IDs
- Vendor/customer information
- Transaction descriptions
- Any unique identifiers

Exclude any encrypted or sensitive personal information columns.
"""
    return prompt

def get_matching_prompt(bank_data, invoice_data, column_info):
    """Create prompt for LLM to match transactions."""
    prompt = f"""
You are a financial reconciliation expert. Please match transactions between bank statement and invoice data based on the following criteria:

**Column Information:**
- Bank key columns: {column_info['bank_key_columns']}
- Invoice key columns: {column_info['invoice_key_columns']}
- Primary match fields - Bank: {column_info['primary_match_fields']['bank']}, Invoice: {column_info['primary_match_fields']['invoice']}
- Secondary match fields - Bank: {column_info['secondary_match_fields']['bank']}, Invoice: {column_info['secondary_match_fields']['invoice']}

**Bank Statement Data (Complete Important Columns):**
{bank_data.to_string()}

**Invoice Data (Complete Important Columns):**
{invoice_data.to_string()}

Please perform transaction matching and return the results in the following JSON format:
{{
    "matches": [
        {{
            "bank_record_index": 0,
            "invoice_record_index": 0,
            "confidence_score": 0.95,
            "match_reason": "Exact amount and date match"
        }}
    ],
    "unmatched_bank_indices": [0, 1, 2],
    "unmatched_invoice_indices": [0, 1, 2],
    "summary": {{
        "total_bank_records": 0,
        "total_invoice_records": 0,
        "matched_pairs": 0,
        "unmatched_bank": 0,
        "unmatched_invoices": 0
    }}
}}

Matching Rules:
1. Primary match: Same amount and similar dates (within 7 days)
2. Secondary match: Similar descriptions/references and amounts
3. Confidence scoring: 0.9+ for exact matches, 0.7+ for probable matches, 0.5+ for possible matches
4. Only include matches with confidence >= 0.7

Important: Return valid JSON only. Do not include any explanatory text outside the JSON structure.
"""
    return prompt

# def call_gemini_api(prompt, max_retries=3):
#     """Call Gemini API with retry logic."""
#     for attempt in range(max_retries):
#         try:
#             response = model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             logger.error(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}")
#             if attempt == max_retries - 1:
#                 raise
#     return None

def call_chatgpt_api(prompt, max_retries=3):
    """Call ChatGPT API with retry logic."""
    for attempt in range(max_retries):
        try:
            # Use the client object for the API call
            response = client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Accessing the content from the response object
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ChatGPT API call failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
    return None

def parse_json_response(response_text):
    """Parse JSON response from LLM, handling potential formatting issues."""
    try:
        # Try to extract JSON from the response
        # Sometimes LLM adds extra text, so we look for JSON structure
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx + 1]
            return json.loads(json_str)
        else:
            # If no clear JSON structure, try the whole response
            return json.loads(response_text)
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.error(f"Response text: {response_text[:500]}...")
        raise ValueError(f"Invalid JSON response from LLM: {str(e)}")

# --- API Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Financial Reconciliation Backend'
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and basic preprocessing."""
    try:
        logger.info("Starting file upload process")
        
        # Check if files are present
        if 'bank_statement' not in request.files or 'invoices' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both bank_statement and invoices files are required'
            }), 400
        
        bank_file = request.files['bank_statement']
        invoice_file = request.files['invoices']
        
        # Validate files
        if bank_file.filename == '' or invoice_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        if not (allowed_file(bank_file.filename) and allowed_file(invoice_file.filename)):
            return jsonify({
                'success': False,
                'error': 'Invalid file format. Allowed formats: CSV, XLSX, XLS'
            }), 400
        
        # Save files
        bank_filename = secure_filename(f"bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bank_file.filename}")
        invoice_filename = secure_filename(f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{invoice_file.filename}")
        
        bank_path = os.path.join(app.config['UPLOAD_FOLDER'], bank_filename)
        invoice_path = os.path.join(app.config['UPLOAD_FOLDER'], invoice_filename)
        
        bank_file.save(bank_path)
        invoice_file.save(invoice_path)
        
        # Read files
        logger.info("Reading files")
        bank_df_raw = read_file(bank_path)
        invoice_df_raw = read_file(invoice_path)
        
        # Basic cleanup only
        bank_df_clean = basic_cleanup(bank_df_raw)
        invoice_df_clean = basic_cleanup(invoice_df_raw)
        
        # Encrypt sensitive data
        bank_df_encrypted, bank_sensitive_cols = encrypt_sensitive_data(bank_df_clean)
        invoice_df_encrypted, invoice_sensitive_cols = encrypt_sensitive_data(invoice_df_clean)
        
        # Store processed data for column identification step
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}")
        bank_df_encrypted.to_pickle(f"{session_file}_bank.pkl")
        invoice_df_encrypted.to_pickle(f"{session_file}_invoice.pkl")
        
        # Prepare response with sample data (first 5 rows for LLM analysis)
        response_data = {
            'success': True,
            'message': 'Files uploaded and basic preprocessing completed successfully',
            'session_id': session_id,
            'preprocessing_info': {
                'bank_original_rows': len(bank_df_raw),
                'bank_processed_rows': len(bank_df_encrypted),
                'invoice_original_rows': len(invoice_df_raw),
                'invoice_processed_rows': len(invoice_df_encrypted),
                'bank_sensitive_columns': bank_sensitive_cols,
                'invoice_sensitive_columns': invoice_sensitive_cols
            },
            'bank_statement_sample': bank_df_encrypted.head().to_dict('records'),
            'invoices_sample': invoice_df_encrypted.head().to_dict('records')
        }
        
        # Clean up uploaded files
        os.remove(bank_path)
        os.remove(invoice_path)
        
        logger.info("File upload and basic preprocessing completed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'File processing failed: {str(e)}'
        }), 500

@app.route('/identify_columns', methods=['POST'])
def identify_columns():
    """Identify key columns for matching using LLM, then preprocess those columns."""
    logger.info("=== Starting identifying columns ===")
    print("ROUTE HIT /identify!") 
    
    # Initialize variables for cleanup in case of errors
    bank_processed_file = None
    invoice_processed_file = None
    column_info_file = None
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID is required'
            }), 400
        
        # Load processed data
        bank_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_bank.pkl")
        invoice_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_invoice.pkl")
        
        # Check if input files exist
        if not os.path.exists(bank_file):
            logger.error(f"Bank file not found: {bank_file}")
            return jsonify({
                'success': False,
                'error': f'Bank data file not found for session {session_id}. Please upload files again.'
            }), 404
            
        if not os.path.exists(invoice_file):
            logger.error(f"Invoice file not found: {invoice_file}")
            return jsonify({
                'success': False,
                'error': f'Invoice data file not found for session {session_id}. Please upload files again.'
            }), 404
        
        # Try to load the pickle files with error handling
        try:
            bank_df = pd.read_pickle(bank_file)
            logger.info(f"Successfully loaded bank data: {bank_df.shape}")
        except Exception as e:
            logger.error(f"Failed to load bank pickle file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load bank data: {str(e)}'
            }), 500
            
        try:
            invoice_df = pd.read_pickle(invoice_file)
            logger.info(f"Successfully loaded invoice data: {invoice_df.shape}")
        except Exception as e:
            logger.error(f"Failed to load invoice pickle file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load invoice data: {str(e)}'
            }), 500
        
        # Validate dataframes are not empty
        if bank_df.empty:
            logger.error("Bank dataframe is empty")
            return jsonify({
                'success': False,
                'error': 'Bank data is empty'
            }), 400
            
        if invoice_df.empty:
            logger.error("Invoice dataframe is empty")
            return jsonify({
                'success': False,
                'error': 'Invoice data is empty'
            }), 400
        
        # Get first 5 rows for LLM analysis
        bank_sample = bank_df.head(5)
        invoice_sample = invoice_df.head(5)
        
        # Call LLM to identify columns
        logger.info("Calling LLM to identify key columns")
        try:
            prompt = get_column_identification_prompt(bank_sample, invoice_sample)
            #llm_response = call_gemini_api(prompt)
            llm_response = call_chatgpt_api(prompt)
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'LLM API call failed: {str(e)}'
            }), 500

        # Parse response
        try:
            column_info = parse_json_response(llm_response)
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to parse LLM response: {str(e)}'
            }), 500
        
        # Validate that identified columns exist in the data
        bank_columns = list(bank_df.columns)
        invoice_columns = list(invoice_df.columns)
        
        # Filter out non-existent columns and log warnings
        original_bank_cols = column_info.get('bank_key_columns', [])
        original_invoice_cols = column_info.get('invoice_key_columns', [])
        
        column_info['bank_key_columns'] = [col for col in original_bank_cols if col in bank_columns]
        column_info['invoice_key_columns'] = [col for col in original_invoice_cols if col in invoice_columns]
        
        # Log any missing columns
        missing_bank_cols = [col for col in original_bank_cols if col not in bank_columns]
        missing_invoice_cols = [col for col in original_invoice_cols if col not in invoice_columns]
        
        if missing_bank_cols:
            logger.warning(f"Bank columns not found in data: {missing_bank_cols}")
        if missing_invoice_cols:
            logger.warning(f"Invoice columns not found in data: {missing_invoice_cols}")
        
        # Check if we have any valid columns to process
        if not column_info['bank_key_columns']:
            logger.error("No valid bank key columns identified")
            return jsonify({
                'success': False,
                'error': 'No valid bank key columns could be identified'
            }), 400
            
        if not column_info['invoice_key_columns']:
            logger.error("No valid invoice key columns identified")
            return jsonify({
                'success': False,
                'error': 'No valid invoice key columns could be identified'
            }), 400
        
        # Now preprocess only the important columns identified by LLM
        logger.info("Preprocessing important columns identified by LLM")
        try:
            bank_df_processed = preprocess_important_columns(bank_df, column_info['bank_key_columns'], "bank statement")
            logger.info(f"Bank preprocessing completed: {bank_df_processed.shape}")
        except Exception as e:
            logger.error(f"Bank preprocessing failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Bank data preprocessing failed: {str(e)}'
            }), 500
            
        try:
            invoice_df_processed = preprocess_important_columns(invoice_df, column_info['invoice_key_columns'], "invoices")
            logger.info(f"Invoice preprocessing completed: {invoice_df_processed.shape}")
        except Exception as e:
            logger.error(f"Invoice preprocessing failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Invoice data preprocessing failed: {str(e)}'
            }), 500
        
        # Validate processed dataframes
        if bank_df_processed.empty:
            logger.error("Processed bank dataframe is empty")
            return jsonify({
                'success': False,
                'error': 'Processed bank data is empty'
            }), 500
            
        if invoice_df_processed.empty:
            logger.error("Processed invoice dataframe is empty")
            return jsonify({
                'success': False,
                'error': 'Processed invoice data is empty'
            }), 500
        
        # Define file paths for saving
        bank_processed_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_bank_processed.pkl")
        invoice_processed_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_invoice_processed.pkl")
        column_info_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_columns.json")
        
        # Save preprocessed data with error handling and verification
        try:
            logger.info(f"Saving processed bank data to: {bank_processed_file}")
            bank_df_processed.to_pickle(bank_processed_file)
            
            # Verify the file was created and is not empty
            if not os.path.exists(bank_processed_file):
                raise Exception(f"Bank processed file was not created: {bank_processed_file}")
            
            file_size = os.path.getsize(bank_processed_file)
            if file_size == 0:
                raise Exception(f"Bank processed file is empty: {bank_processed_file}")
            
            logger.info(f"✓ Bank processed file created successfully: {bank_processed_file} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save processed bank data: {str(e)}")
            # Clean up any partial files
            if os.path.exists(bank_processed_file):
                os.remove(bank_processed_file)
            return jsonify({
                'success': False,
                'error': f'Failed to save processed bank data: {str(e)}'
            }), 500
        
        try:
            logger.info(f"Saving processed invoice data to: {invoice_processed_file}")
            invoice_df_processed.to_pickle(invoice_processed_file)
            
            # Verify the file was created and is not empty
            if not os.path.exists(invoice_processed_file):
                raise Exception(f"Invoice processed file was not created: {invoice_processed_file}")
            
            file_size = os.path.getsize(invoice_processed_file)
            if file_size == 0:
                raise Exception(f"Invoice processed file is empty: {invoice_processed_file}")
            
            logger.info(f"✓ Invoice processed file created successfully: {invoice_processed_file} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save processed invoice data: {str(e)}")
            # Clean up any partial files
            if os.path.exists(invoice_processed_file):
                os.remove(invoice_processed_file)
            if os.path.exists(bank_processed_file):
                os.remove(bank_processed_file)
            return jsonify({
                'success': False,
                'error': f'Failed to save processed invoice data: {str(e)}'
            }), 500
        
        # Save column info with error handling
        try:
            logger.info(f"Saving column info to: {column_info_file}")
            with open(column_info_file, 'w') as f:
                json.dump(column_info, f, indent=2)
            
            # Verify the file was created and is not empty
            if not os.path.exists(column_info_file):
                raise Exception(f"Column info file was not created: {column_info_file}")
                
            file_size = os.path.getsize(column_info_file)
            if file_size == 0:
                raise Exception(f"Column info file is empty: {column_info_file}")
            
            logger.info(f"✓ Column info file created successfully: {column_info_file} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save column info: {str(e)}")
            # Clean up any partial files
            for file_path in [column_info_file, bank_processed_file, invoice_processed_file]:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            return jsonify({
                'success': False,
                'error': f'Failed to save column information: {str(e)}'
            }), 500
        
        # Final verification - try to read back the files to ensure they're valid
        try:
            # Test reading back the pickle files
            test_bank_df = pd.read_pickle(bank_processed_file)
            test_invoice_df = pd.read_pickle(invoice_processed_file)
            
            # Test reading back the JSON file
            with open(column_info_file, 'r') as f:
                test_column_info = json.load(f)
                
            logger.info("✓ All files successfully verified by reading them back")
            
        except Exception as e:
            logger.error(f"File verification failed: {str(e)}")
            # Clean up files that can't be read
            for file_path in [column_info_file, bank_processed_file, invoice_processed_file]:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            return jsonify({
                'success': False,
                'error': f'File verification failed - files may be corrupted: {str(e)}'
            }), 500
        
        logger.info("Column identification and preprocessing completed successfully")
        
        # Create detailed response with file status
        response_data = {
            'success': True,
            'column_info': column_info,
            'message': 'Key columns identified and preprocessed successfully',
            'preprocessing_summary': {
                'bank_key_columns_processed': column_info['bank_key_columns'],
                'invoice_key_columns_processed': column_info['invoice_key_columns'],
                'bank_rows_processed': len(bank_df_processed),
                'invoice_rows_processed': len(invoice_df_processed)
            },
            'files_created': {
                'bank_processed_file': {
                    'path': bank_processed_file,
                    'size_bytes': os.path.getsize(bank_processed_file),
                    'exists': True
                },
                'invoice_processed_file': {
                    'path': invoice_processed_file,
                    'size_bytes': os.path.getsize(invoice_processed_file),
                    'exists': True
                },
                'column_info_file': {
                    'path': column_info_file,
                    'size_bytes': os.path.getsize(column_info_file),
                    'exists': True
                }
            }
        }
        
        # Add warnings if any columns were missing
        if missing_bank_cols or missing_invoice_cols:
            response_data['warnings'] = {
                'missing_bank_columns': missing_bank_cols,
                'missing_invoice_columns': missing_invoice_cols
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in identify_columns: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up any partially created files
        cleanup_files = [bank_processed_file, invoice_processed_file, column_info_file]
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up partial file: {file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up file {file_path}: {cleanup_error}")
        
        return jsonify({
            'success': False,
            'error': f'Column identification failed: {str(e)}',
            'files_created': {
                'bank_processed_file': {'exists': False},
                'invoice_processed_file': {'exists': False},
                'column_info_file': {'exists': False}
            }
        }), 500
    
@app.route('/match', methods=['POST'])
def match_transactions():
    """Perform AI-powered transaction matching using complete preprocessed important columns."""
    logger.info("=== Starting transaction matching process ===")
    print("ROUTE HIT!") 
    
    try:
        # Step 1: Get request data
        logger.info("Step 1: Extracting request data")
        try:
            data = request.get_json()
            logger.info(f"Request data received: {data}")
        except Exception as e:
            logger.error(f"Error parsing JSON request: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Invalid JSON in request body'
            }), 400
        
        # Step 2: Validate session ID
        logger.info("Step 2: Validating session ID")
        session_id = data.get('session_id')
        if not session_id:
            logger.error("Session ID is missing from request")
            return jsonify({
                'success': False,
                'error': 'Session ID is required'
            }), 400
        
        logger.info(f"Session ID: {session_id}")
        
        # Step 3: Construct file paths
        logger.info("Step 3: Constructing file paths")
        try:
            bank_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_bank_processed.pkl")
            invoice_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_invoice_processed.pkl")
            column_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_columns.json")
            
            logger.info(f"Bank file path: {bank_file}")
            logger.info(f"Invoice file path: {invoice_file}")
            logger.info(f"Column file path: {column_file}")
        except Exception as e:
            logger.error(f"Error constructing file paths: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error constructing file paths'
            }), 500

        # Step 4: Check file existence
        logger.info("Step 4: Checking file existence")
        missing_files = []
        for file_path in [bank_file, invoice_file, column_file]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                logger.error(f"Missing file: {file_path}")
            else:
                logger.info(f"File exists: {file_path}")
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return jsonify({
                'success': False,
                'error': f'Session data not found. Missing files: {missing_files}. Please complete previous steps.'
            }), 404
        
        # Step 5: Load data files
        logger.info("Step 5: Loading data files")
        
        # Load bank data
        try:
            logger.info("Loading bank data...")
            bank_df_full = pd.read_pickle(bank_file)
            logger.info(f"Bank data loaded successfully. Shape: {bank_df_full.shape}")
            logger.info(f"Bank data columns: {list(bank_df_full.columns)}")
        except Exception as e:
            logger.error(f"Error loading bank data: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error loading bank data: {str(e)}'
            }), 500
        
        # Load invoice data
        try:
            logger.info("Loading invoice data...")
            invoice_df_full = pd.read_pickle(invoice_file)
            logger.info(f"Invoice data loaded successfully. Shape: {invoice_df_full.shape}")
            logger.info(f"Invoice data columns: {list(invoice_df_full.columns)}")
        except Exception as e:
            logger.error(f"Error loading invoice data: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error loading invoice data: {str(e)}'
            }), 500
        
        # Load column info
        try:
            logger.info("Loading column info...")
            with open(column_file, 'r') as f:
                column_info = json.load(f)
            logger.info(f"Column info loaded successfully: {column_info}")
        except Exception as e:
            logger.error(f"Error loading column info: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error loading column info: {str(e)}'
            }), 500
        
        # Step 6: Extract important columns
        logger.info("Step 6: Extracting important columns")
        
        try:
            # Validate column info structure
            if 'bank_key_columns' not in column_info:
                logger.error("bank_key_columns not found in column_info")
                return jsonify({
                    'success': False,
                    'error': 'bank_key_columns not found in column info'
                }), 500
            
            if 'invoice_key_columns' not in column_info:
                logger.error("invoice_key_columns not found in column_info")
                return jsonify({
                    'success': False,
                    'error': 'invoice_key_columns not found in column info'
                }), 500
            
            logger.info(f"Bank key columns: {column_info['bank_key_columns']}")
            logger.info(f"Invoice key columns: {column_info['invoice_key_columns']}")
            
            # Check if key columns exist in dataframes
            missing_bank_cols = [col for col in column_info['bank_key_columns'] if col not in bank_df_full.columns]
            missing_invoice_cols = [col for col in column_info['invoice_key_columns'] if col not in invoice_df_full.columns]
            
            if missing_bank_cols:
                logger.error(f"Missing bank columns: {missing_bank_cols}")
                return jsonify({
                    'success': False,
                    'error': f'Missing bank columns: {missing_bank_cols}'
                }), 500
            
            if missing_invoice_cols:
                logger.error(f"Missing invoice columns: {missing_invoice_cols}")
                return jsonify({
                    'success': False,
                    'error': f'Missing invoice columns: {missing_invoice_cols}'
                }), 500
            
            # Extract important columns
            bank_important_data = bank_df_full[column_info['bank_key_columns']].copy()
            invoice_important_data = invoice_df_full[column_info['invoice_key_columns']].copy()
            
            logger.info(f"Bank important data shape: {bank_important_data.shape}")
            logger.info(f"Invoice important data shape: {invoice_important_data.shape}")
            logger.info(f"Bank important data sample:\n{bank_important_data.head()}")
            logger.info(f"Invoice important data sample:\n{invoice_important_data.head()}")
            
        except Exception as e:
            logger.error(f"Error extracting important columns: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error extracting important columns: {str(e)}'
            }), 500
        
        # Step 7: Prepare batching
        logger.info("Step 7: Preparing data for LLM processing")
        
        try:
            logger.info(f"Sending complete important columns data to LLM for matching")
            logger.info(f"Bank records: {len(bank_important_data)}, Invoice records: {len(invoice_important_data)}")
            
            max_records_per_batch = 200  # Adjust based on API limits
            
            if len(bank_important_data) > max_records_per_batch or len(invoice_important_data) > max_records_per_batch:
                # Process in batches for large datasets
                bank_batch = bank_important_data.head(max_records_per_batch)
                invoice_batch = invoice_important_data.head(max_records_per_batch)
                logger.info(f"Processing in batches due to large dataset size")
                logger.info(f"Bank batch size: {len(bank_batch)}, Invoice batch size: {len(invoice_batch)}")
            else:
                # Process complete dataset
                bank_batch = bank_important_data
                invoice_batch = invoice_important_data
                logger.info(f"Processing complete dataset")
            
            logger.info(f"Final batch sizes - Bank: {len(bank_batch)}, Invoice: {len(invoice_batch)}")
            
        except Exception as e:
            logger.error(f"Error preparing batches: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error preparing data batches: {str(e)}'
            }), 500
        
        # Step 8: Generate prompt and call LLM
        logger.info("Step 8: Calling LLM for matching")
        
        try:
            logger.info("Generating matching prompt...")
            prompt = get_matching_prompt(bank_batch, invoice_batch, column_info)
            logger.info(f"Prompt generated successfully (length: {len(prompt)} characters)")
            logger.info(f"Prompt preview: {prompt[:500]}...")
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error generating prompt: {str(e)}'
            }), 500
        
        try:
            logger.info("Calling ChatGPT API...")
            llm_response = call_chatgpt_api(prompt)
            logger.info(f"LLM response received (length: {len(str(llm_response))} characters)")
            logger.info(f"LLM response preview: {str(llm_response)[:500]}...")
            
        except Exception as e:
            logger.error(f"Error calling ChatGPT API: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error calling ChatGPT API: {str(e)}'
            }), 500
        
        # Step 9: Parse matching results
        logger.info("Step 9: Parsing matching results")
        
        try:
            logger.info("Parsing JSON response...")
            matching_results = parse_json_response(llm_response)
            logger.info(f"Matching results parsed successfully")
            logger.info(f"Matching results structure: {type(matching_results)}")
            logger.info(f"Matching results keys: {list(matching_results.keys()) if isinstance(matching_results, dict) else 'Not a dict'}")
            
            if isinstance(matching_results, dict) and 'matches' in matching_results:
                logger.info(f"Number of matches found: {len(matching_results['matches'])}")
            else:
                logger.warning(f"Unexpected matching results structure: {matching_results}")
                
        except Exception as e:
            logger.error(f"Error parsing matching results: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error parsing matching results: {str(e)}'
            }), 500
        
        # Step 10: Enhance results with full record data
        logger.info("Step 10: Enhancing results with full record data")
        
        try:
            enhanced_matches = []
            matches = matching_results.get('matches', [])
            logger.info(f"Processing {len(matches)} matches for enhancement")
            
            for i, match in enumerate(matches):
                try:
                    logger.info(f"Processing match {i+1}/{len(matches)}")
                    
                    if 'bank_record_index' not in match:
                        logger.error(f"Match {i+1} missing bank_record_index: {match}")
                        continue
                    
                    if 'invoice_record_index' not in match:
                        logger.error(f"Match {i+1} missing invoice_record_index: {match}")
                        continue
                    
                    bank_idx = match['bank_record_index']
                    invoice_idx = match['invoice_record_index']
                    
                    logger.info(f"Match {i+1} - Bank index: {bank_idx}, Invoice index: {invoice_idx}")
                    
                    # Validate indices
                    if bank_idx >= len(bank_df_full):
                        logger.error(f"Bank index {bank_idx} out of range (max: {len(bank_df_full)-1})")
                        continue
                    
                    if invoice_idx >= len(invoice_df_full):
                        logger.error(f"Invoice index {invoice_idx} out of range (max: {len(invoice_df_full)-1})")
                        continue
                    
                    # Get full records from original data
                    bank_full_record = bank_df_full.iloc[bank_idx].to_dict()
                    invoice_full_record = invoice_df_full.iloc[invoice_idx].to_dict()
                    
                    enhanced_match = {
                        'file_a_entry': bank_full_record,
                        'file_b_entry': invoice_full_record,
                        'confidence_score': match.get('confidence_score', 0),
                        'match_reason': match.get('match_reason', 'No reason provided')
                    }
                    enhanced_matches.append(enhanced_match)
                    logger.info(f"Match {i+1} enhanced successfully")
                    
                except Exception as e:
                    logger.error(f"Error enhancing match {i+1}: {str(e)}")
                    logger.error(f"Problematic match data: {match}")
                    continue
            
            logger.info(f"Enhanced {len(enhanced_matches)} matches successfully")
            
        except Exception as e:
            logger.error(f"Error enhancing matches: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error enhancing matches: {str(e)}'
            }), 500
        
        # Step 11: Get unmatched records
        logger.info("Step 11: Identifying unmatched records")
        
        try:
            matched_bank_indices = [m['bank_record_index'] for m in matching_results.get('matches', []) if 'bank_record_index' in m]
            matched_invoice_indices = [m['invoice_record_index'] for m in matching_results.get('matches', []) if 'invoice_record_index' in m]
            
            logger.info(f"Matched bank indices: {matched_bank_indices}")
            logger.info(f"Matched invoice indices: {matched_invoice_indices}")
            
            unmatched_bank = bank_df_full[~bank_df_full.index.isin(matched_bank_indices)].to_dict('records')
            unmatched_invoices = invoice_df_full[~invoice_df_full.index.isin(matched_invoice_indices)].to_dict('records')
            
            logger.info(f"Unmatched bank records: {len(unmatched_bank)}")
            logger.info(f"Unmatched invoice records: {len(unmatched_invoices)}")
            
        except Exception as e:
            logger.error(f"Error identifying unmatched records: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error identifying unmatched records: {str(e)}'
            }), 500
        
        # Step 12: Prepare final response
        logger.info("Step 12: Preparing final response")
        
        try:
            final_results = {
                'success': True,
                'message': f'Matching completed. Found {len(enhanced_matches)} matches.',
                'matches': enhanced_matches,
                'unmatched_file_a_entries': unmatched_bank,
                'unmatched_file_b_entries': unmatched_invoices,
                'summary': {
                    'total_bank_records': len(bank_df_full),
                    'total_invoice_records': len(invoice_df_full),
                    'matched_pairs': len(enhanced_matches),
                    'unmatched_bank': len(unmatched_bank),
                    'unmatched_invoices': len(unmatched_invoices)
                },
                'column_info': column_info
            }
            
            logger.info(f"Final results summary: {final_results['summary']}")
            
        except Exception as e:
            logger.error(f"Error preparing final response: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error preparing final response: {str(e)}'
            }), 500
        
        # Step 13: Clean up session files
        logger.info("Step 13: Cleaning up session files")
        
        try:
            session_files = [
                f"{session_id}_bank.pkl",
                f"{session_id}_invoice.pkl",
                f"{session_id}_bank_processed.pkl",
                f"{session_id}_invoice_processed.pkl",
                f"{session_id}_columns.json"
            ]
            
            cleaned_files = []
            for filename in session_files:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleaned_files.append(filename)
                        logger.info(f"Cleaned up file: {filename}")
                    else:
                        logger.info(f"File not found for cleanup: {filename}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {filename}: {str(e)}")
            
            logger.info(f"Cleaned up {len(cleaned_files)} files: {cleaned_files}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Don't fail the request for cleanup errors
        
        logger.info("=== Transaction matching completed successfully ===")
        return jsonify(final_results)
        
    except Exception as e:
        logger.error(f"=== CRITICAL ERROR in match_transactions ===")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Transaction matching failed: {str(e)}'
        }), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Financial Reconciliation Backend on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)