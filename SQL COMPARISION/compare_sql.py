import sqlparse
import pandas as pd
import os
import re
import csv

# --- Helper functions ---
def clean_sql_line(line):
    """Remove inline comments and extra spaces."""
    line = re.sub(r'--.*', '', line)
    line = re.sub(r'/\*.*?\*/', '', line)
    return line.strip()

def normalize_datatype(dt):
    dt = dt.upper()
    dt = dt.replace('INTEGER', 'INT')
    dt = dt.replace('CHARACTER VARYING', 'VARCHAR')
    dt = dt.replace('NUMERIC', 'DECIMAL')
    dt = re.sub(r'\s+', ' ', dt)
    return dt

# --- Parse SQL ---
def parse_sql_file(file_path):
    with open(file_path, 'r') as f:
        sql = f.read()
    statements = sqlparse.split(sql)
    db_structure = {}
    for stmt in statements:
        parsed = sqlparse.parse(stmt)[0]
        tokens = [t.value for t in parsed.tokens if not t.is_whitespace]
        if tokens and tokens[0].upper() == 'CREATE' and 'TABLE' in tokens[1].upper():
            table_name = None
            for t in tokens:
                t_upper = t.upper()
                if t_upper not in ['CREATE','TABLE','IF','NOT','EXISTS','`']:
                    table_name = t.replace('`','')
                    break
            if not table_name:
                continue
            columns_part = stmt[stmt.find('(')+1:stmt.rfind(')')]
            columns_lines = [clean_sql_line(line) for line in columns_part.split(',') if line.strip()]
            columns_dict = {}
            for col in columns_lines:
                col_upper = col.upper()
                if col_upper.startswith('PRIMARY') or col_upper.startswith('FOREIGN') or col_upper.startswith('UNIQUE'):
                    continue
                parts = col.split()
                if len(parts) >= 2:
                    col_name = parts[0].replace('`','')
                    data_type = normalize_datatype(parts[1])
                    not_null = 'NOT NULL' in col_upper
                    default_val = None
                    match = re.search(r'DEFAULT\s+([\S]+)', col_upper)
                    if match:
                        default_val = match.group(1)
                    columns_dict[col_name] = {'datatype': data_type, 'not_null': not_null, 'default': default_val}
            db_structure[table_name] = columns_dict
    return db_structure

# --- Compare databases ---
def compare_databases(db1_struct, db2_struct):
    report = []
    all_tables = set(db1_struct.keys()).union(set(db2_struct.keys()))
    missing_tables = missing_columns = changed_columns = 0
    for table in all_tables:
        db1_cols = db1_struct.get(table, {})
        db2_cols = db2_struct.get(table, {})
        if table not in db1_struct:
            report.append([table, 'Missing in db1','','','',''])
            missing_tables += 1
            continue
        if table not in db2_struct:
            report.append([table, 'Missing in db2','','','',''])
            missing_tables += 1
            continue
        all_cols = set(db1_cols.keys()).union(set(db2_cols.keys()))
        for col in all_cols:
            col1 = db1_cols.get(col)
            col2 = db2_cols.get(col)
            if col1 is None:
                report.append([table,'Column missing in db1',col,col2['datatype'],col2['not_null'],col2['default']])
                missing_columns +=1
            elif col2 is None:
                report.append([table,'Column missing in db2',col,col1['datatype'],col1['not_null'],col1['default']])
                missing_columns +=1
            else:
                changes=[]
                if col1['datatype'] != col2['datatype']:
                    changes.append(f"Type: {col1['datatype']} -> {col2['datatype']}")
                if col1['not_null'] != col2['not_null']:
                    changes.append(f"NotNull: {col1['not_null']} -> {col2['not_null']}")
                if col1['default'] != col2['default']:
                    changes.append(f"Default: {col1['default']} -> {col2['default']}")
                if changes:
                    report.append([table,'Column changed',col,', '.join(changes),col1['datatype'],col2['datatype']])
                    changed_columns +=1
    return pd.DataFrame(report,columns=['Table','Issue','Column','Details','DB1 Info','DB2 Info']), missing_tables, missing_columns, changed_columns

# --- Main ---
if __name__=="__main__":
    db1_file='db1.sql'
    db2_file='db2.sql'
    db1_struct=parse_sql_file(db1_file)
    db2_struct=parse_sql_file(db2_file)
    df_report, missing_tables, missing_columns, changed_columns = compare_databases(db1_struct,db2_struct)

    # Ensure CSV-safe
    df_report = df_report.fillna('')
    for col in ['Details','DB1 Info','DB2 Info']:
        df_report[col]=df_report[col].astype(str)
        if col=='Details':
            df_report[col]=df_report[col].apply(lambda x: re.sub(r'\((.*?)\)', lambda m: '(' + m.group(1).replace(',',';') + ')', x))

    # Create directories if not exist
    os.makedirs('reports',exist_ok=True)
    os.makedirs('summary',exist_ok=True)

    # Write summary CSV
    summary_file='summary/db_comparison_summary.csv'
    with open(summary_file,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerow(['Summary'])
        writer.writerow(['Missing Tables',missing_tables])
        writer.writerow(['Missing Columns',missing_columns])
        writer.writerow(['Changed Columns',changed_columns])

    # Write detailed report CSV
    report_file='reports/db_comparison_report.csv'
    with open(report_file,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerow(['Table','Issue','Column','Details','DB1 Info','DB2 Info'])
        for row in df_report.itertuples(index=False):
            writer.writerow(row)

    print(f"Summary CSV generated: {summary_file}")
    print(f"Detailed report CSV generated: {report_file}")
