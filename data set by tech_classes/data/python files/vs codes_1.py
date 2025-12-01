import pandas as pd
import os
import logging
import time
from sqlalchemy import create_engine


logging.basicConfig(
    filename='data_log/ingestion.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
CHUNK_SIZE = 100000 
HUGE_FILE_NAME = 'sales.csv' 
# ---------------------

# 1. DATABASE CONNECTION DETAILS
db_user = 'root'
db_password = '8520147'
db_host = 'localhost'
db_port = 3306
db_name = 'tech_classes'

# 2. FILE PATH
file_path = r"C:\Users\ACER\OneDrive\Desktop\my_projects\mysql+python+power bi project_by_tech classes\data set by tech_classes\data"

# 3. CREATE ENGINE
engine_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
print(engine_url)
engine = create_engine(engine_url)


def ingest_db(df, table_name, engine, if_exists_value='replace'):
    """Inserts a DataFrame (or a chunk) into the MySQL database."""
    # Use 'replace' for single files or the first chunk. Use 'append' for subsequent chunks.
    df.to_sql(table_name, con=engine, if_exists=if_exists_value, index=False)


## 4. HYBRID FILE LISTING AND INGESTION
print("\n--- Starting Hybrid Data Ingestion ---")
# 

def load_raw_data():
    """ This function will load the CSVs as dataframe and ingest into db """
    start = time.time()
    for file in os.listdir(file_path): 
        if file.endswith('.csv'):
            
            full_file_path = os.path.join(file_path, file)
            table_name = file[:-4]
            
            # --- A. CHUNKED MODE FOR HUGE FILE ---
            if file == HUGE_FILE_NAME:
                print(f"\n[CHUNK MODE] Starting load for {file} (Large Data)...")
                
                # The Pandas reader returns an iterable object instead of a single DataFrame
                chunk_iterator = pd.read_csv(full_file_path, chunksize=CHUNK_SIZE)
                
                # Set the initial mode to REPLACE, then switch to APPEND
                if_exists_mode = 'replace' 
                
                for i, chunk in enumerate(chunk_iterator):
                    print(f"   -> Inserting chunk #{i+1}, rows {CHUNK_SIZE*i + 1} to {CHUNK_SIZE*(i+1) + 1}")
                    
                    # Ingest the chunk
                    ingest_db(chunk, table_name, engine, if_exists_mode)

                    # After the first chunk, switch to appending the rest of the data
                    if_exists_mode = 'append' 
                    
                print(f"-> {table_name} ingestion complete via {i+1} chunks.")

            # --- B. DIRECT MODE FOR SMALLER FILES ---
            else:
                print(f"\n[DIRECT MODE] Starting load for {file} (Small Data)...")
                
                # Read the entire file into memory (safe for smaller files)
                df = pd.read_csv(full_file_path)
                
                print(f"   -> Read {file}, Shape: {df.shape}")
                
                # Ingest the entire file, replacing the table if it exists
                ingest_db(df, table_name, engine, if_exists_value='replace') 
                
                print(f"-> {table_name} ingestion complete in one step.")

    end = time.time()
    total_time_taken = (end - start)/60
    logging.info('=====Ingestion Completed=====')
    logging.info(f'\n total_time_taken : {total_time_taken:.2f} minutes')


if __name__ == '__main__' :
    print("\n--- Starting Hybrid Data Ingestion ---")
    load_raw_data()

