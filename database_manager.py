import sqlite3
import json
import numpy as np
import os

DB_FILE = 'electre_problems.db' # Define default DB name globally or pass it around

def connect_db(db_file=DB_FILE):
    """Connects to the SQLite database, creating it if it doesn't exist."""
    # Ensure the directory exists if db_file includes a path
    db_dir = os.path.dirname(db_file)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign key constraints
    return conn

def create_schema(conn):
    """Creates the necessary tables if they don't exist."""
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS criteria (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        type TEXT NOT NULL CHECK(type IN ('cost', 'benefit')),
        weight REAL NOT NULL,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
        UNIQUE (problem_id, name) -- Criterion name must be unique within a problem
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alternatives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
        UNIQUE (problem_id, name) -- Alternative name must be unique within a problem
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scores (
        alternative_id INTEGER NOT NULL,
        criterion_id INTEGER NOT NULL,
        score REAL NOT NULL,
        FOREIGN KEY (alternative_id) REFERENCES alternatives(id) ON DELETE CASCADE,
        FOREIGN KEY (criterion_id) REFERENCES criteria(id) ON DELETE CASCADE,
        PRIMARY KEY (alternative_id, criterion_id)
    );
    """)
    conn.commit()

def reset_database(db_file=DB_FILE):
    """Drops all existing tables and recreates the schema for a fresh start."""
    conn = None
    try:
        print(f"Attempting to reset database: {db_file}")
        conn = connect_db(db_file)
        cursor = conn.cursor()
        print("Dropping existing tables (if they exist)...")
        cursor.execute("DROP TABLE IF EXISTS scores;")
        cursor.execute("DROP TABLE IF EXISTS alternatives;")
        cursor.execute("DROP TABLE IF EXISTS criteria;")
        cursor.execute("DROP TABLE IF EXISTS problems;")
        print("Tables dropped.")
        # Recreate the schema
        print("Recreating schema...")
        create_schema(conn) # create_schema already commits
        print(f"Database '{db_file}' has been reset successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred during database reset: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed after reset.")

def load_json_data(json_file='input_data.json'):
    """Loads data from the specified JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Basic validation
        if 'problem_name' not in data or 'criteria' not in data or 'alternatives' not in data:
            raise ValueError("JSON missing required keys: 'problem_name', 'criteria', 'alternatives'")
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file}")
        return None
    except ValueError as ve:
        print(f"Error: Invalid JSON structure: {ve}")
        return None

def populate_database(conn, problem_data):
    """Populates the database with data for a specific problem from parsed JSON."""
    cursor = conn.cursor()
    problem_name = problem_data['problem_name']

    try:
        # --- Get or Create Problem --- 
        cursor.execute("SELECT id FROM problems WHERE name = ?", (problem_name,))
        problem_row = cursor.fetchone()
        if problem_row:
            problem_id = problem_row['id']
            # Clear old data for this problem to prevent inconsistencies on re-run
            print(f"Problem '{problem_name}' exists (ID: {problem_id}). Clearing old associated data...")
            cursor.execute("DELETE FROM scores WHERE criterion_id IN (SELECT id FROM criteria WHERE problem_id = ?)", (problem_id,))
            cursor.execute("DELETE FROM scores WHERE alternative_id IN (SELECT id FROM alternatives WHERE problem_id = ?)", (problem_id,))
            cursor.execute("DELETE FROM criteria WHERE problem_id = ?", (problem_id,))
            cursor.execute("DELETE FROM alternatives WHERE problem_id = ?", (problem_id,))
        else:
            cursor.execute("INSERT INTO problems (name) VALUES (?) RETURNING id", (problem_name,))
            problem_id = cursor.fetchone()['id']
            print(f"Created new problem '{problem_name}' with ID: {problem_id}")

        # --- Insert Criteria --- 
        criteria_map = {} # name -> id
        for criterion in problem_data['criteria']:
            cursor.execute("""
            INSERT INTO criteria (problem_id, name, type, weight) 
            VALUES (?, ?, ?, ?) RETURNING id
            """, (problem_id, criterion['name'], criterion['type'], criterion['weight']))
            criterion_id = cursor.fetchone()['id']
            criteria_map[criterion['name']] = criterion_id
        print(f"Inserted {len(criteria_map)} criteria.")

        # --- Insert Alternatives & Scores --- 
        alternative_map = {} # name -> id
        scores_inserted = 0
        for alternative in problem_data['alternatives']:
            cursor.execute("""
            INSERT INTO alternatives (problem_id, name) 
            VALUES (?, ?) RETURNING id
            """, (problem_id, alternative['name']))
            alternative_id = cursor.fetchone()['id']
            alternative_map[alternative['name']] = alternative_id

            # Insert scores for this alternative
            for crit_name, score_value in alternative['scores'].items():
                if crit_name not in criteria_map:
                    print(f"Warning: Criterion '{crit_name}' mentioned in scores for alternative '{alternative['name']}' not found in criteria list. Skipping score.")
                    continue
                criterion_id = criteria_map[crit_name]
                cursor.execute("""
                INSERT INTO scores (alternative_id, criterion_id, score) 
                VALUES (?, ?, ?)
                """, (alternative_id, criterion_id, score_value))
                scores_inserted += 1
        print(f"Inserted {len(alternative_map)} alternatives and {scores_inserted} scores.")

        conn.commit()
        print("Database populated successfully.")
        return problem_id

    except sqlite3.IntegrityError as e:
        print(f"Database Error: {e}. Rolling back changes.")
        conn.rollback()
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Rolling back changes.")
        conn.rollback()
        return None

def get_data_for_solver(conn, problem_name):
    """Retrieves data for a given problem name and formats it for ElectreSolver."""
    cursor = conn.cursor()
    
    # --- Get Problem ID --- 
    cursor.execute("SELECT id FROM problems WHERE name = ?", (problem_name,))
    problem_row = cursor.fetchone()
    if not problem_row:
        print(f"Error: Problem '{problem_name}' not found in the database.")
        return None
    problem_id = problem_row['id']
    
    # --- Get Criteria (in a consistent order) ---
    cursor.execute("SELECT id, name, type, weight FROM criteria WHERE problem_id = ? ORDER BY id", (problem_id,))
    criteria_rows = cursor.fetchall()
    if not criteria_rows:
        print(f"Error: No criteria found for problem '{problem_name}'.")
        return None
        
    criteria_ids = [row['id'] for row in criteria_rows]
    criteria_names = [row['name'] for row in criteria_rows]
    criteria_types = [row['type'] for row in criteria_rows]
    weights = np.array([row['weight'] for row in criteria_rows])
    criterion_id_to_index = {cid: i for i, cid in enumerate(criteria_ids)}

    # --- Get Alternatives (in a consistent order) ---
    cursor.execute("SELECT id, name FROM alternatives WHERE problem_id = ? ORDER BY id", (problem_id,))
    alternative_rows = cursor.fetchall()
    if not alternative_rows:
        print(f"Error: No alternatives found for problem '{problem_name}'.")
        return None
        
    alternative_ids = [row['id'] for row in alternative_rows]
    alternative_names = [row['name'] for row in alternative_rows]
    alternative_id_to_index = {aid: i for i, aid in enumerate(alternative_ids)}
    
    # --- Get Scores and build the decision matrix ---
    num_alts = len(alternative_names)
    num_crits = len(criteria_names)
    decision_matrix = np.zeros((num_alts, num_crits))
    
    cursor.execute("""
    SELECT alternative_id, criterion_id, score 
    FROM scores 
    WHERE alternative_id IN (SELECT id FROM alternatives WHERE problem_id = ?)
    """, (problem_id,))
    score_rows = cursor.fetchall()
    
    scores_found = 0
    for row in score_rows:
        alt_id = row['alternative_id']
        crit_id = row['criterion_id']
        score = row['score']
        
        if alt_id in alternative_id_to_index and crit_id in criterion_id_to_index:
            row_idx = alternative_id_to_index[alt_id]
            col_idx = criterion_id_to_index[crit_id]
            decision_matrix[row_idx, col_idx] = score
            scores_found += 1
        
    # Optional check: Ensure we found scores for all matrix cells (or handle missing scores)
    expected_scores = num_alts * num_crits
    if scores_found != expected_scores:
         print(f"Warning: Found {scores_found} scores, but expected {expected_scores}. Matrix might have zeros for missing scores.")

    return {
        'decision_matrix': decision_matrix,
        'weights': weights,
        'criteria_types': criteria_types,
        'alternative_names': alternative_names,
        'criteria_names': criteria_names
    }

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    JSON_FILE = 'input_data.json'

    # <<< Reset the database before populating (optional) >>>
    reset_database(DB_FILE) 
    # <<< Remove or comment out the line above if you don't want to reset each time >>>

    # 1. Load data from JSON
    print(f"\nLoading data from {JSON_FILE}...")
    problem_data = load_json_data(JSON_FILE)

    if problem_data:
        # 2. Connect to DB and create schema if needed
        conn = connect_db(DB_FILE)
        create_schema(conn)

        # 3. Populate DB from JSON data
        problem_id = populate_database(conn, problem_data)

        if problem_id:
            # 4. Retrieve data formatted for the solver
            solver_data = get_data_for_solver(conn, problem_data['problem_name'])
            
            if solver_data:
                print("\n--- Data Retrieved for Solver ---")
                print("Alternative Names:", solver_data['alternative_names'])
                print("Criteria Names:", solver_data['criteria_names'])
                print("Weights:", solver_data['weights'])
                print("Criteria Types:", solver_data['criteria_types'])
                print("Decision Matrix:\n", solver_data['decision_matrix'])
        
        # Close the connection
        conn.close()
        print("\nDatabase connection closed.") 