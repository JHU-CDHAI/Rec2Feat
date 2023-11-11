import json
import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from functools import reduce
import sqlite3
from collections import OrderedDict
from recfldgrn.datapoint import convert_PID_to_PIDgroup
from fastparquet import ParquetFile

class CRFTC_Base(Dataset):
    # Constructor
    def __init__(self, RANGE_SIZE, CaseDB_Path, CASE_CACHE_SIZE):
        # Essential attributes
        self.CRFCT_RANGE_SIZE = RANGE_SIZE
        self.CaseDB_Path = CaseDB_Path
        
        # Attributes for handling the dataset
        self.df_PDT_all = None  # DataFrame for all data
        self.LastDataset = None
        self.NameCRFTC = None  # Name of the column/feature to be used
        
        # Cache to store recently accessed cases
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()  # Using OrderedDict for LRU cache
        
        
    def get_cache_case(self, index):
        # Retrieve a case from the main DataFrame using the given index
        Case = self.df_PDT_all.iloc[index].copy()
        pid, preddt = Case['PID'], Case['PredDT']
        
        # Check if the case is in the cache
        ValueCRFTC = self.Cache_Store.get((pid, preddt))
         
        # If found in cache, return it; otherwise, return None
        if ValueCRFTC is not None:
            Case[self.NameCRFTC] = ValueCRFTC
        else:
            Case = None
        return Case
        
    def add_to_cache(self, Case):
        # Add a new case to the cache
        pid, preddt = Case['PID'], Case['PredDT']
        NameCRFTC = self.NameCRFTC
        ValueCRFTC = Case[NameCRFTC]
        cache_key = (pid, preddt)

        # Update cache and maintain its size
        
        if len(self.Cache_Store) >= self.cache_size:
            self.Cache_Store.popitem(last=False)
        # elif cache_key in self.Cache_Store:
        #     self.Cache_Store.move_to_end(cache_key)
            
        self.Cache_Store[cache_key] = ValueCRFTC
        
    def get_bucket_case(self, index):
        # Retrieve a specific case based on the given index from the dataframe 'df_PDT_all'
        Case = self.df_PDT_all.iloc[index].copy()
        pid, preddt = Case['PID'], Case['PredDT']  # Extracting PID and PredDT from the case

        # Retrieve the name of the CRFTC field from the class instance
        NameCRFTC = self.NameCRFTC

        # Determine the group based on PID and set file paths for the bucket and its info
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        bucket_path = os.path.join(bucket_directory, Group + '.parquet')
        bucket_info_path = os.path.join(bucket_directory, Group + '_info.parquet')
    
        if not os.path.exists(bucket_info_path) and not os.path.exists(bucket_path):
            Case = None; return Case
            
        # Read the bucket information file to find the matching case_info
        info = pd.read_parquet(bucket_info_path, engine='fastparquet', index=False)
        case_info = info[(info['PID'] == pid) & (info['PredDT'] == preddt)]
        
        if not case_info.empty:
            CRFTCValue_nrow = case_info.iloc[0]['nrow']
            if CRFTCValue_nrow > 0:
                CRFTCValue = pd.DataFrame()
                parquet_file = ParquetFile(bucket_path)
                for df in parquet_file.iter_row_groups():
                    matched_df = df[(df['PID'] == pid) & (df['PredDT'] == preddt)]
                    if matched_df.empty: continue
                    CRFTCValue = pd.concat([CRFTCValue, matched_df], ignore_index=True)
                    if len(CRFTCValue) == CRFTCValue_nrow: break
            else:
                CRFTCValue = None
            Case[NameCRFTC] = CRFTCValue
        else:
            # Set Case to None if no matching row is found
            Case = None
        return Case
    
    def add_to_bucket(self, Case):
        # Extract PID and PredDT from the given case
        Case = Case.copy()
        pid, preddt = Case['PID'], Case['PredDT']
        # Retrieve the name of the CRFTC field from the class instance
        NameCRFTC = self.NameCRFTC

        # Determine the group based on PID and set file paths for the bucket and its info
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        bucket_path = os.path.join(bucket_directory, Group + '.parquet')
        bucket_info_path = os.path.join(bucket_directory, Group + '_info.parquet')

        # Create the bucket directory if it doesn't exist
        os.makedirs(bucket_directory, exist_ok=True)

        # Extract the CRFTC value from the case
        CRFTCValue = Case[NameCRFTC]

        # Determine the number of rows in CRFTCValue
        nrow = len(CRFTCValue) if CRFTCValue is not None else 0

        # Create a DataFrame for the number of rows information
        df_nrow = pd.DataFrame([{'PID': pid, 'PredDT': preddt, 'nrow': nrow}])
        # Add or update the bucket info file with the number of rows information
        if os.path.exists(bucket_info_path):
            df_nrow.to_parquet(bucket_info_path, engine='fastparquet', index=False, append=True)
        else:
            df_nrow.to_parquet(bucket_info_path, engine='fastparquet', index=False)

        # If CRFTCValue is None, end the function here
        if CRFTCValue is None: return None

        # Add or update the bucket file with the CRFTCValue
        if os.path.exists(bucket_path):
            CRFTCValue.to_parquet(bucket_path, engine='fastparquet', index=False, append=True)
        else:
            CRFTCValue.to_parquet(bucket_path, engine='fastparquet', index=False)
    

    def get_db_case(self, index):
        # Copy the case data from the main DataFrame using the provided index
        Case = self.df_PDT_all.iloc[index].copy()
        pid, preddt = Case['PID'], Case['PredDT']
        NameCRFTC = self.NameCRFTC

        # Determine the group and set file paths for the database and info
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, self.NameCRFTC)
        db_path = os.path.join(bucket_directory, Group + '.db')
        info_path = os.path.join(bucket_directory, 'info.db')

        # Return None if the info file does not exist
        if not os.path.exists(info_path): return None

        # Load case_info from the pickle file
        conn = sqlite3.connect(info_path)
        info_query = f'''SELECT * FROM "{NameCRFTC}" WHERE PID = ? AND PredDT = ?'''
        df_case_info = pd.read_sql_query(info_query, conn, params=(pid, str(preddt)))
        conn.close()
        
        # Check if the case exists in the case_info DataFrame
        # matched_case = df_case_info[(df_case_info['PID'] == pid) & (df_case_info['PredDT'] == preddt)]
        if df_case_info.empty: Case = None; return Case
        
        if df_case_info.iloc[0]['nrow'] == 0: Case[NameCRFTC] = None; return Case

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        NameCRFTC = self.NameCRFTC

        # Prepare and execute the query to fetch CRFTC values for the case
        CRFTC_query = f'''SELECT * FROM "{NameCRFTC}" WHERE PID = ? AND PredDT = ?'''
        CRFTCValue = pd.read_sql_query(CRFTC_query, conn, params=(pid, str(preddt)))
        
        DT_cols = [i for i in CRFTCValue.columns if 'DT' in i]
        for DT_col in DT_cols: CRFTCValue[DT_col] = pd.to_datetime(CRFTCValue[DT_col])
        
        tkn_cols = [i for i in CRFTCValue.columns if 'Grn_' in i]
        for tkn_col in tkn_cols: CRFTCValue[tkn_col] = CRFTCValue[tkn_col].apply(lambda x: json.loads(x))
        
        Case[NameCRFTC] = CRFTCValue
        # Close the database connection
        conn.close()
        return Case
    
    def add_to_db(self, Case):
        Case = Case.copy()
        pid, preddt = Case['PID'], str(Case['PredDT'])

        # Retrieve the name of the CRFTC field from the class instance
        NameCRFTC = self.NameCRFTC

        # Determine the number of rows in CRFTCValue
        nrow = len(Case[NameCRFTC]) if Case[NameCRFTC] is not None else 0

        # Determine the group and set file paths for the database and info
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        db_path = os.path.join(bucket_directory, Group + '.db')
        info_path = os.path.join(bucket_directory, 'info.db')

        # Create the bucket directory if it doesn't exist
        os.makedirs(bucket_directory, exist_ok=True)
        
        # check whether the case is in info_path or not.
        conn = sqlite3.connect(info_path)
        info_query = f'''SELECT * FROM "{NameCRFTC}" WHERE PID = ? AND PredDT = ?'''
        try:
            df_case_info = pd.read_sql_query(info_query, conn, params=(pid, str(preddt)))
        except:
            df_case_info = pd.DataFrame(columns = ['PID', 'PredDT', 'nrow'])
        
        # len(df_case_info) > 0, which means we have it already.
        if len(df_case_info) > 0: conn.close(); return None 
        
        # len(df_case_info) == 0, here we need to update the database.
        # 1. update info_path
        df_new = pd.DataFrame([{'PID': pid, 'PredDT': str(preddt), 'nrow': nrow}])
        df_new.to_sql(NameCRFTC, conn, if_exists='append', index=False)

        cursor = conn.cursor()
        create_index_query = f'''CREATE INDEX IF NOT EXISTS idx_pid_preddt ON "{self.NameCRFTC}" (PID, PredDT);'''
        cursor.execute(create_index_query)
        conn.commit()
        conn.close()
    
        # 2. update db_path
        # Return None if CRFTCValue is None
        if Case[NameCRFTC] is None: return None

        CRFTCValue = Case[NameCRFTC].copy()

        # Connect to the SQLite database and update CRFTCValue
        conn = sqlite3.connect(db_path)
        # print(CRFTCValue.columns)
        assert 'PID' in CRFTCValue.columns and 'PredDT' in CRFTCValue.columns
        assert len(CRFTCValue) > 0

        tkn_cols = [i for i in CRFTCValue.columns if 'Grn_' in i]
        for tkn_col in tkn_cols: CRFTCValue[tkn_col] = CRFTCValue[tkn_col].apply(lambda x: json.dumps(x))
        
        CRFTCValue.to_sql(NameCRFTC, conn, if_exists='append', index=False)

        cursor = conn.cursor()
        create_index_query = f'''CREATE INDEX IF NOT EXISTS idx_pid_preddt ON "{self.NameCRFTC}" (PID, PredDT);'''
        cursor.execute(create_index_query)
        conn.commit()
        conn.close()

    
    def __getitem__(self, index):
        Case = self.get_cache_case(index)
        if Case is not None: return Case
    
        Case = self.get_bucket_case(index)
        if Case is not None: 
            self.add_to_cache(Case)
            return Case
        
        # Case = self.get_db_case(index)
        # if Case is not None: 
        #     self.add_to_cache(Case)
        #     return Case
    
        Case = self.execute_case(index)
        self.add_to_cache(Case)
        # self.add_to_bucket(Case)
        self.add_to_db(Case)
        return Case
    
    def __len__(self):
        return len(self.df_PDT_all)
    
