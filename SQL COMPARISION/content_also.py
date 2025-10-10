import pandas as pd
import mysql.connector  # For MySQL
import csv
import os

# --- Connect to databases ---
db1_conn = mysql.connector.connect(
    host="localhost",
    user="user1",
    password="Sh!kherjain1",
    database="db1"
)

db2_conn = mysql.connector.connect(
    host="localhost",
    user="user1",
    password="Sh!kherjain1",
    database="db2"
)

db1_cursor = db1_conn.cursor(dictionary=True)
db2_cursor = db2_conn.cursor(dictionary=True)

# --- Helper to get table list ---
def get_tables(cursor):
    cursor.execute("SHOW TABLES;")
    return [row[f'Tables_in_{cursor.connection.database}'] for row in cursor.fetchall()]

# --- Helper to get columns and schema ---
def get_table_schema(cursor, table):
    cursor.execute(f"DESCRIBE {table};")
    return {row['Field']: {'type': row['Type'], 'null': row['Null'], 'key': row['Key'], 'default': row['Default']} for row in cursor.fetchall()}

# --- Helper to fetch table content ---
def get_table_content(cursor, table, order_by=None):
    order_clause = f"ORDER BY {order_by}" if order_by else ""
    cursor.execute(f"SELECT * FROM {table} {order_clause};")
    return cursor.fetchall()

# --- Compare databases ---
report = []
summary = {'Missing Tables':0, 'Missing Columns':0, 'Changed Columns':0, 'Content Differences':0}

tables_db1 = set(get_tables(db1_cursor))
tables_db2 = set(get_tables(db2_cursor))

all_tables = tables_db1.union(tables_db2)

for table in all_tables:
    if table not in tables_db1:
        report.append([table,'Missing in db1','','','',''])
        summary['Missing Tables'] += 1
        continue
    if table not in tables_db2:
        report.append([table,'Missing in db2','','','',''])
        summary['Missing Tables'] += 1
        continue

    # Compare schema
    schema1 = get_table_schema(db1_cursor, table)
    schema2 = get_table_schema(db2_cursor, table)

    all_columns = set(schema1.keys()).union(set(schema2.keys()))
    for col in all_columns:
        col1 = schema1.get(col)
        col2 = schema2.get(col)
        if col1 is None:
            report.append([table,'Column missing in db1',col,col2['type'],col2['null'],col2['default']])
            summary['Missing Columns'] += 1
        elif col2 is None:
            report.append([table,'Column missing in db2',col,col1['type'],col1['null'],col1['default']])
            summary['Missing Columns'] += 1
        else:
            changes=[]
            if col1['type'] != col2['type']:
                changes.append(f"Type: {col1['type']} -> {col2['type']}")
            if col1['null'] != col2['null']:
                changes.append(f"Null: {col1['null']} -> {col2['null']}")
            if col1['default'] != col2['default']:
                changes.append(f"Default: {col1['default']} -> {col2['default']}")
            if changes:
                report.append([table,'Column changed',col,', '.join(changes),col1['type'],col2['type']])
                summary['Changed Columns'] += 1

    # Compare content (row by row)
    # Must have a primary key or unique column to order consistently
    primary_keys = [c for c,v in schema1.items() if v['key']=='PRI']
    order_col = primary_keys[0] if primary_keys else None
    rows1 = get_table_content(db1_cursor, table, order_col)
    rows2 = get_table_content(db2_cursor, table, order_col)

    # Quick row count check
    if len(rows1) != len(rows2):
        report.append([table,'Content row count differs', '', f'{len(rows1)} rows in db1 -> {len(rows2)} rows in db2','',''])
        summary['Content Differences'] += 1
    else:
        # Compare each row
        for r1, r2 in zip(rows1, rows2):
            diffs = []
            for col in r1.keys():
                v1 = r1[col]
                v2 = r2.get(col)
                if v1 != v2:
                    diffs.append(f"{col}: {v1} -> {v2}")
            if diffs:
                report.append([table,'Content differs','','; '.join(diffs),'',''])
                summary['Content Differences'] += 1

# --- Write CSVs ---
os.makedirs('reportsC', exist_ok=True)
os.makedirs('summaryC', exist_ok=True)

# Detailed report
report_file = 'reportsC/db_comparison_report.csv'
with open(report_file,'w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['Table','Issue','Column','Details','DB1 Info','DB2 Info'])
    for row in report:
        writer.writerow(row)

# Summary
summary_file = 'summaryC/db_comparison_summary.csv'
with open(summary_file,'w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['Metric','Count'])
    for k,v in summary.items():
        writer.writerow([k,v])

print(f"Summary CSV: {summary_file}")
print(f"Detailed report CSV: {report_file}")
