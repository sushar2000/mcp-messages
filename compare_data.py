# Script to compare KKChat table with KKChat.tsv file row by row
import pyodbc
import argparse
import json
import os
from datetime import datetime
from collections import defaultdict

# ANSI color codes for terminal output


class Colors:
    GREEN = '\033[92m'     # Green for 100% matches
    YELLOW = '\033[93m'    # Yellow for 90-99% matches
    # Cyan for numbers (same as command line text params)
    CYAN = '\033[96m'
    RED = '\033[91m'       # Red for errors
    RESET = '\033[0m'      # Reset to normal color


def load_config(config_file='config.json'):
    """Load database configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        print("Please create a config.json file with database connection details")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return None


# Load configuration
config = None
server = None
database = None
user = None
password = None
table_name = None
conn_str = None


def initialize_config(config_file='config.json'):
    """Initialize global configuration variables"""
    global config, server, database, user, password, table_name, conn_str

    config = load_config(config_file)
    if not config:
        return False

    # Database connection details from config
    db_config = config.get('database', {})
    server = db_config.get('host')
    port = db_config.get('port')
    database = db_config.get('dbname')
    user = db_config.get('user')
    password = db_config.get('password')
    table_name = db_config.get('table_name')

    conn_str = (
        f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};'
        f'UID={user};PWD={password};'
    )
    return True


def read_tsv_file(file_path):
    """Read TSV file and return list of tuples (date, sender, message)"""
    tsv_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    date_str = parts[0]
                    sender = parts[1]
                    # Join remaining parts in case message contains tabs
                    message = '\t'.join(parts[2:])
                    tsv_data.append((line_num, date_str, sender, message))
                else:
                    print(
                        f"Warning: Line {line_num} has only {len(parts)} columns: {line}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return []

    return tsv_data


def read_db_table():
    """Read KKChat table and return list of tuples (date, sender, message)"""
    db_data = []
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            query = f"SELECT DateColumn, SenderColumn, MessageColumn FROM {table_name} ORDER BY DateColumn, SenderColumn, MessageColumn"
            cursor.execute(query)
            for row in cursor:
                date_obj = row[0]
                sender = row[1]
                message = row[2]

                # Convert datetime to string for comparison (use specific format: YYYY-MM-DD HH:MM:SS)
                if isinstance(date_obj, datetime):
                    date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_str = str(date_obj) if date_obj else ''

                db_data.append((date_str, sender or '', message or ''))
    except Exception as e:
        print(f"Error reading database: {e}")
        return []

    return db_data


def normalize_date(date_str):
    """Normalize date string for comparison - convert to format: YYYY-MM-DD HH:MM:SS"""
    if not date_str:
        return ''

    # Try different date formats and convert to standard format (including AM/PM support)
    formats = [
        '%Y-%m-%d %H:%M:%S',      # YYYY-MM-DD HH:MM:SS
        '%m/%d/%Y %H:%M:%S',      # MM/DD/YYYY HH:MM:SS
        '%d/%m/%Y %H:%M:%S',      # DD/MM/YYYY HH:MM:SS
        '%Y-%m-%d %I:%M:%S %p',   # YYYY-MM-DD HH:MM:SS AM/PM
        '%m/%d/%Y %I:%M:%S %p',   # MM/DD/YYYY HH:MM:SS AM/PM
        '%d/%m/%Y %I:%M:%S %p',   # DD/MM/YYYY HH:MM:SS AM/PM
        '%Y-%m-%d %I:%M %p',      # YYYY-MM-DD HH:MM AM/PM
        '%m/%d/%Y %I:%M %p',      # MM/DD/YYYY HH:MM AM/PM
        '%d/%m/%Y %I:%M %p',      # DD/MM/YYYY HH:MM AM/PM
        '%Y-%m-%d',               # YYYY-MM-DD (add time)
        '%m/%d/%Y',               # MM/DD/YYYY (add time)
        '%d/%m/%Y',               # DD/MM/YYYY (add time)
        '%Y-%m-%d %H:%M',         # YYYY-MM-DD HH:MM (add seconds)
        '%m/%d/%Y %H:%M',         # MM/DD/YYYY HH:MM (add seconds)
        '%d/%m/%Y %H:%M'          # DD/MM/YYYY HH:MM (add seconds)
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            # Always return in 24-hour format: YYYY-MM-DD HH:MM:SS
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

    # If no format matches, return as is
    return date_str.strip()


def create_comparison_key(date_str, sender, message):
    """Create a normalized key for comparison"""
    norm_date = normalize_date(date_str)
    norm_sender = (sender or '').strip()
    norm_message = (message or '').strip()
    return (norm_date, norm_sender, norm_message)


def compare_data(tsv_file_path, verbose=False, limit=None):
    """Compare TSV file with database table"""
    print("Reading TSV file...")
    tsv_data = read_tsv_file(tsv_file_path)

    print("Reading database table...")
    db_data = read_db_table()

    if not tsv_data:
        print("No data found in file")
        return

    if not db_data:
        print("No data found in database")
        return

    # Check if verbose is a string (with sub-options) or boolean
    show_details = verbose and verbose != False
    show_all_matches = verbose == 'all'
    # Don't show matching rows in 'brief' mode
    show_matching_rows = verbose in ['standard', 'all']

    # Debug output to see what verbose contains
    print(
        f"DEBUG: verbose = '{verbose}', show_details = {show_details}, show_all_matches = {show_all_matches}, show_matching_rows = {show_matching_rows}")

    # Apply limit if specified
    if limit and limit > 0:
        original_tsv_count = len(tsv_data)
        original_db_count = len(db_data)
        tsv_data = tsv_data[:limit]
        db_data = db_data[:limit]
        print(
            f"\nLimit applied: Comparing first {Colors.CYAN}{limit}{Colors.RESET} rows")
        print(
            f"Original TSV rows: {Colors.CYAN}{original_tsv_count}{Colors.RESET}, Limited to: {Colors.CYAN}{len(tsv_data)}{Colors.RESET}")
        print(
            f"Original DB rows: {Colors.CYAN}{original_db_count}{Colors.RESET}, Limited to: {Colors.CYAN}{len(db_data)}{Colors.RESET}")

    print("\nData Summary:")
    print(f"TSV file rows: {Colors.CYAN}{len(tsv_data)}{Colors.RESET}")
    print(f"Database rows: {Colors.CYAN}{len(db_data)}{Colors.RESET}")

    # Create sets for comparison
    tsv_keys = set()
    tsv_map = {}
    tsv_duplicates = 0

    for line_num, date_str, sender, message in tsv_data:
        key = create_comparison_key(date_str, sender, message)
        if key in tsv_keys:
            tsv_duplicates += 1
        tsv_keys.add(key)
        if key not in tsv_map:
            tsv_map[key] = []
        tsv_map[key].append((line_num, date_str, sender, message))

    db_keys = set()
    db_map = {}
    db_duplicates = 0

    for date_str, sender, message in db_data:
        key = create_comparison_key(date_str, sender, message)
        if key in db_keys:
            db_duplicates += 1
        db_keys.add(key)
        if key not in db_map:
            db_map[key] = []
        db_map[key].append((date_str, sender, message))

    print(
        f"TSV unique rows: {Colors.CYAN}{len(tsv_keys)}{Colors.RESET} (duplicates found: {Colors.CYAN}{tsv_duplicates}{Colors.RESET})")
    print(
        f"Database unique rows: {Colors.CYAN}{len(db_keys)}{Colors.RESET} (duplicates found: {Colors.CYAN}{db_duplicates}{Colors.RESET})")

    # Show duplicate rows in verbose mode
    if show_details and (tsv_duplicates > 0 or db_duplicates > 0):
        print("\n=== DUPLICATE ROWS FOUND ===")

        if tsv_duplicates > 0:
            print(
                f"\nTSV Duplicates ({Colors.CYAN}{tsv_duplicates}{Colors.RESET} found):")
            for key, rows in tsv_map.items():
                if len(rows) > 1:
                    print(f"  Duplicate key: {key}")
                    for i, (line_num, date_str, sender, message) in enumerate(rows, 1):
                        print(
                            f"    {Colors.CYAN}{i}{Colors.RESET}. Line {Colors.CYAN}{line_num}{Colors.RESET}: {date_str} | {sender} | {message[:50]}{'...' if len(message) > 50 else ''}")
                    print()

        if db_duplicates > 0:
            print(
                f"\nDatabase Duplicates ({Colors.CYAN}{db_duplicates}{Colors.RESET} found):")
            for key, rows in db_map.items():
                if len(rows) > 1:
                    print(f"  Duplicate key: {key}")
                    for i, (date_str, sender, message) in enumerate(rows, 1):
                        print(
                            f"    {Colors.CYAN}{i}{Colors.RESET}. DB row: {date_str} | {sender} | {message[:50]}{'...' if len(message) > 50 else ''}")
                    print()

    # Find matches, missing, and extras
    matching_keys = tsv_keys & db_keys
    missing_in_db = tsv_keys - db_keys
    extra_in_db = db_keys - tsv_keys

    print("\n=== COMPARISON RESULTS ===")
    print(f"Matching rows: {Colors.CYAN}{len(matching_keys)}{Colors.RESET}")
    print(
        f"Rows in TSV but missing from DB: {Colors.CYAN}{len(missing_in_db)}{Colors.RESET}")
    print(
        f"Rows in DB but not in TSV: {Colors.CYAN}{len(extra_in_db)}{Colors.RESET}")

    # Show missing rows from TSV only in verbose mode
    if missing_in_db and show_details:
        print(
            f"\n=== ROWS MISSING FROM DATABASE ({Colors.CYAN}{len(missing_in_db)}{Colors.RESET} rows) ===")
        for i, key in enumerate(sorted(missing_in_db), 1):
            tsv_rows = tsv_map[key]
            for line_num, date_str, sender, message in tsv_rows:
                print(
                    f"{Colors.CYAN}{i}{Colors.RESET}. Line {Colors.CYAN}{line_num}{Colors.RESET} in TSV:")
                print(f"   Date: {date_str}")
                print(f"   Sender: {sender}")
                print(
                    f"   Message: {message[:100]}{'...' if len(message) > 100 else ''}")
                print()

    # Show extra rows in DB only in verbose mode
    if extra_in_db and show_details:
        print(
            f"\n=== EXTRA ROWS IN DATABASE ({Colors.CYAN}{len(extra_in_db)}{Colors.RESET} rows) ===")
        for i, key in enumerate(sorted(extra_in_db), 1):
            db_rows = db_map[key]
            for date_str, sender, message in db_rows:
                print(f"{Colors.CYAN}{i}{Colors.RESET}. Extra in DB:")
                print(f"   Date: {date_str}")
                print(f"   Sender: {sender}")
                print(
                    f"   Message: {message[:100]}{'...' if len(message) > 100 else ''}")
                print()

    # Show matching rows - sample or all based on verbose level (skip in brief mode)
    if matching_keys and show_details and show_matching_rows:
        if show_all_matches:
            print(
                f"\n=== ALL MATCHING ROWS ({Colors.CYAN}{len(matching_keys)}{Colors.RESET} rows) ===")
            for i, key in enumerate(sorted(matching_keys), 1):
                tsv_rows = tsv_map[key]
                print(f"{Colors.CYAN}{i}{Colors.RESET}. Matching row:")
                print(f"   Date: {key[0]}")
                print(f"   Sender: {key[1]}")
                print(
                    f"   Message: {key[2][:100]}{'...' if len(key[2]) > 100 else ''}")
                print()
        else:
            print(
                f"\n=== SAMPLE MATCHING ROWS (showing first {Colors.CYAN}5{Colors.RESET}) ===")
            for i, key in enumerate(sorted(list(matching_keys)[:5]), 1):
                tsv_rows = tsv_map[key]
                print(f"{Colors.CYAN}{i}{Colors.RESET}. Matching row:")
                print(f"   Date: {key[0]}")
                print(f"   Sender: {key[1]}")
                print(
                    f"   Message: {key[2][:100]}{'...' if len(key[2]) > 100 else ''}")
                print()

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(
        f"Total rows processed - TSV: {Colors.CYAN}{len(tsv_data)}{Colors.RESET}, DB: {Colors.CYAN}{len(db_data)}{Colors.RESET}")
    print(
        f"Unique rows after deduplication - TSV: {Colors.CYAN}{len(tsv_keys)}{Colors.RESET}, DB: {Colors.CYAN}{len(db_keys)}{Colors.RESET}")
    print(
        f"Duplicates found - TSV: {Colors.CYAN}{tsv_duplicates}{Colors.RESET}, DB: {Colors.CYAN}{db_duplicates}{Colors.RESET}")
    print(
        f"Matching unique rows: {Colors.CYAN}{len(matching_keys)}{Colors.RESET}")
    print(f"Missing from DB: {Colors.CYAN}{len(missing_in_db)}{Colors.RESET}")
    print(f"Extra in DB: {Colors.CYAN}{len(extra_in_db)}{Colors.RESET}")

    if len(tsv_keys) > 0:
        match_percentage = (len(matching_keys) / len(tsv_keys)) * 100
        if match_percentage == 100.0:
            # Print in green if 100% match
            print(
                f"Match percentage (based on unique rows): {Colors.GREEN}{match_percentage:.2f}%{Colors.RESET}")
        elif match_percentage >= 90.0:
            # Print in yellow if 90-99% match
            print(
                f"Match percentage (based on unique rows): {Colors.YELLOW}{match_percentage:.2f}%{Colors.RESET}")
        else:
            # Print in cyan for other percentages
            print(
                f"Match percentage (based on unique rows): {Colors.RED}{match_percentage:.2f}%{Colors.RESET}")

    if not show_details and (missing_in_db or extra_in_db):
        print("\nNote: Use --verbose or -v flag to see detailed row-by-row comparison")
        print("      Options: -v brief (no matching rows), -v standard (sample), -v all (all matching rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare KKChat table with all TSV files from config')
    parser.add_argument('-v', '--verbose', nargs='?', const='brief', default=False,
                        choices=['brief', 'standard', 'all'],
                        help='Show detailed row-by-row comparison information. Options: brief (no matching rows), standard (sample), all (all matching rows)')
    parser.add_argument('-l', '--limit', type=int, default=None,
                        help='Limit the number of rows to compare (default: compare all rows)')
    parser.add_argument('-c', '--config', default='config.json',
                        help='Path to configuration file (default: config.json)')
    args = parser.parse_args()

    # Initialize configuration
    if not initialize_config(args.config):
        exit(1)

    # Get all files from config
    file_paths = config.get('file_paths', [])

    if not file_paths:
        print("Error: No file paths found in config file")
        print("Please add 'file_paths' array to your config.json")
        exit(1)

    print(f"Processing {len(file_paths)} files from config...")
    for i, tsv_file_path in enumerate(file_paths, 1):
        print(
            f"\n{Colors.CYAN}=== Processing file {i}/{len(file_paths)}: {tsv_file_path} ==={Colors.RESET}")
        try:
            compare_data(tsv_file_path=tsv_file_path,
                         verbose=args.verbose, limit=args.limit)
        except Exception as e:
            print(f"Error processing {tsv_file_path}: {e}")
            import traceback
            traceback.print_exc()
