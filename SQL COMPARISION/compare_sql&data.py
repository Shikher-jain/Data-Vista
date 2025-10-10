import pandas as pd
import mysql.connector
import csv
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv('db.env')
'''
DB_USER=root
DB_PASSWORD=**************
DB_HOST=localhost
'''
# --- Connect to databases ---
def connect_to_db(host, user, password, database):
    """Establishes a connection to the specified MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            buffered=True
        )
        print(f"Successfully connected to {database} at {host}")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to {database} at {host}: {err}")
        return None

# --- Helper to get a list of all tables in a database ---
def get_tables(cursor):
    """
    Fetches a list of all tables from the connected database.
    Uses the private '_connection' attribute to get the database name.
    """
    try:
        cursor.execute("SHOW TABLES;")
        db_name = cursor._connection.database
        return [row[f'Tables_in_{db_name}'] for row in cursor.fetchall()]
    except mysql.connector.Error as err:
        print(f"Error getting tables: {err}")
        return []

# --- Helper to get the schema for a single table ---
def get_table_schema(cursor, table):
    """
    Fetches the schema of a table.
    Returns a dictionary of column details.
    """
    try:
        cursor.execute(f"DESCRIBE {table};")
        return {row['Field']: {'type': row['Type'], 'null': row['Null'], 'key': row['Key'], 'default': row['Default']} for row in cursor.fetchall()}
    except mysql.connector.Error as err:
        print(f"Error getting schema for table {table}: {err}")
        return {}

# --- Helper to fetch the content of a single table ---
def get_table_content(cursor, table, order_by=None):
    """
    Fetches all content from a table, optionally ordered by a column.
    """
    try:
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        cursor.execute(f"SELECT * FROM {table} {order_clause};")
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error getting content for table {table}: {err}")
        return []

# --- Main function to compare two databases ---
def compare_databases(db1_cursor, db2_cursor):
    """Compares the schema and content of two databases and returns a report."""
    report = []
    summary = {'Missing Tables': 0, 'Missing Columns': 0, 'Changed Columns': 0, 'Content Differences': 0}

    tables_db1 = set(get_tables(db1_cursor))
    tables_db2 = set(get_tables(db2_cursor))

    all_tables = tables_db1.union(tables_db2)

    for table in all_tables:
        print(f"Comparing table: {table}")
        
        if table not in tables_db1:
            report.append([table, 'Missing in db1', '', '', '', ''])
            summary['Missing Tables'] += 1
            print(f"Table {table} is missing in db1.")
            continue
        if table not in tables_db2:
            report.append([table, 'Missing in db2', '', '', '', ''])
            summary['Missing Tables'] += 1
            print(f"Table {table} is missing in db2.")
            continue

        schema1 = get_table_schema(db1_cursor, table)
        schema2 = get_table_schema(db2_cursor, table)
        
        all_columns = set(schema1.keys()).union(set(schema2.keys()))
        for col in sorted(all_columns):
            col1 = schema1.get(col)
            col2 = schema2.get(col)

            if col1 is None:
                report.append([table, 'Column missing in db1', col, col2['type'], col2['null'], col2['default']])
                summary['Missing Columns'] += 1
                print(f"Column '{col}' is missing in db1's table '{table}'.")
            elif col2 is None:
                report.append([table, 'Column missing in db2', col, col1['type'], col1['null'], col1['default']])
                summary['Missing Columns'] += 1
                print(f"Column '{col}' is missing in db2's table '{table}'.")
            else:
                changes = []
                if col1['type'] != col2['type']:
                    changes.append(f"Type: {col1['type']} -> {col2['type']}")
                if col1['null'] != col2['null']:
                    changes.append(f"Null: {col1['null']} -> {col2['null']}")
                if col1['default'] != col2['default']:
                    changes.append(f"Default: {col1['default']} -> {col2['default']}")
                if changes:
                    report.append([table, 'Column changed', col, ', '.join(changes), col1['type'], col2['type']])
                    summary['Changed Columns'] += 1
                    print(f"Column '{col}' in table '{table}' has changed: {', '.join(changes)}")

        primary_keys = [c for c, v in schema1.items() if v['key'] == 'PRI']
        order_col = primary_keys[0] if primary_keys else None
        
        rows1 = get_table_content(db1_cursor, table, order_col)
        rows2 = get_table_content(db2_cursor, table, order_col)

        if len(rows1) != len(rows2):
            report.append([table, 'Content row count differs', '', f'{len(rows1)} rows in db1 -> {len(rows2)} rows in db2', '', ''])
            summary['Content Differences'] += 1
            print(f"Table '{table}' has different row counts: {len(rows1)} in db1 vs {len(rows2)} in db2.")
        else:
            for r1, r2 in zip(rows1, rows2):
                diffs = []
                pk_val = r1.get(order_col, '') if order_col else ''
                
                for col in r1.keys():
                    v1 = r1[col]
                    v2 = r2.get(col)
                    
                    if v1 != v2:
                        diffs.append(f"{col}: {v1} -> {v2}")
                
                if diffs:
                    report.append([table, 'Content differs', pk_val, '; '.join(diffs), '', ''])
                    summary['Content Differences'] += 1
                    print(f"Content difference found in table '{table}' at row with key '{pk_val}'.")

    return report, summary

# --- Write reports to CSV files ---
def write_report_to_csv(report, filename):
    """Writes a list of lists to a CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(report)
    print(f"Report written to {filename}")

# --- Main execution block ---
if __name__ == "__main__":
    db1_conn = None
    db2_conn = None
    
    # Get credentials from environment variables
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    if not DB_PASSWORD:
        print("Error: DB_PASSWORD environment variable is not set. Please create a .env file name db.env.")
    else:
        db1_conn = connect_to_db(DB_HOST, DB_USER, DB_PASSWORD, "db1")
        db2_conn = connect_to_db(DB_HOST, DB_USER, DB_PASSWORD, "db2")
        
        if db1_conn and db2_conn:
            db1_cursor = db1_conn.cursor(dictionary=True)
            db2_cursor = db2_conn.cursor(dictionary=True)

            report, summary = compare_databases(db1_cursor, db2_cursor)

            report_data = [['Table', 'Issue', 'Column/PK', 'Details', 'DB1 Info', 'DB2 Info']] + report
            write_report_to_csv(report_data, 'reports/db_comparison_report_with_content.csv')

            summary_data = [['Metric', 'Count']] + list(summary.items())
            write_report_to_csv(summary_data, 'summary/db_comparison_summary_with_content.csv')

            print("Comparison complete. Reports generated.")
        else:
            print("Failed to connect to one or both databases. Please check the connection details.")

    if db1_conn and db1_conn.is_connected():
        db1_conn.close()
        print("db1 connection closed.")
    if db2_conn and db2_conn.is_connected():
        db2_conn.close()
        print("db2 connection closed.")
