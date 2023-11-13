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
# from fastparquet import ParquetFile


class CRFTC_Base(Dataset):
    # Constructor
    def __init__(self, 
                 CaseDB_Path, 
                 CRFTC_RANGE_SIZE, 
                 CASE_CACHE_SIZE,
                 use_cache = False,
                 use_db = True, **kwargs):
        
        # Essential attributes
        self.use_db = use_db
        self.use_cache = use_cache
        self.CaseDB_Path = CaseDB_Path
        
        # Attributes for handling the dataset
        self.df_PDT_all = None  # DataFrame for all data
        self.LastDataset = None
        self.NameCRFTC = None  # Name of the column/feature to be used
        self.CRFTC_RANGE_SIZE = CRFTC_RANGE_SIZE
        
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
        if len(self.Cache_Store) >= self.cache_size:
            self.Cache_Store.popitem(last=False)
        self.Cache_Store[cache_key] = ValueCRFTC
         
    def get_db_case(self, index):
        # Copy the case data from the main DataFrame using the provided index
        Case = self.df_PDT_all.iloc[index].copy()
        pid, preddt = Case['PID'], Case['PredDT']
        NameCRFTC = self.NameCRFTC

        # Determine the group and set file paths for the database and info
        Group = convert_PID_to_PIDgroup(pid, self.CRFTC_RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, self.NameCRFTC)
        db_path = os.path.join(bucket_directory, Group + '.db')
        info_path = os.path.join(bucket_directory, 'info.db')

        # Return None if the info file does not exist
        if not os.path.exists(info_path): return None

        # Load case_info from the pickle file
        conn = sqlite3.connect(info_path)
        try:
            info_query = f'''SELECT * FROM "Table" WHERE PID = ? AND PredDT = ?'''
            df_case_info = pd.read_sql_query(info_query, conn, params=(pid, str(preddt)))
        except:
            df_case_info = pd.DataFrame()
        conn.close()
        
        # Check if the case exists in the case_info DataFrame
        # matched_case = df_case_info[(df_case_info['PID'] == pid) & (df_case_info['PredDT'] == preddt)]
        if df_case_info.empty: Case = None; return Case
        
        if df_case_info.iloc[0]['nrow'] == 0: Case[NameCRFTC] = None; return Case

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        NameCRFTC = self.NameCRFTC
        try:
            # Prepare and execute the query to fetch CRFTC values for the case
            CRFTC_query = f'''SELECT * FROM "Table" WHERE PID = ? AND PredDT = ?'''
            CRFTCValue = pd.read_sql_query(CRFTC_query, conn, params=(pid, str(preddt)))
        except:
            CRFTCValue = pd.DataFrame()
        conn.close()
        
        if CRFTCValue.empty: 
            print(f'--- error in loading data frame db for: {NameCRFTC}')
            Case = None; return Case
        
        DT_cols = [i for i in CRFTCValue.columns if 'DT' in i]
        for DT_col in DT_cols: CRFTCValue[DT_col] = pd.to_datetime(CRFTCValue[DT_col])
        
        tkn_cols = [i for i in CRFTCValue.columns if 'Grn_' in i]
        for tkn_col in tkn_cols: CRFTCValue[tkn_col] = CRFTCValue[tkn_col].apply(lambda x: json.loads(x))
        
        Case[NameCRFTC] = CRFTCValue
        # Close the database connection
        
        return Case
    
    def add_to_db(self, Case):
        Case = Case.copy()
        pid, preddt = Case['PID'], str(Case['PredDT'])

        # Retrieve the name of the CRFTC field from the class instance
        NameCRFTC = self.NameCRFTC

        # Determine the number of rows in CRFTCValue
        nrow = len(Case[NameCRFTC]) if Case[NameCRFTC] is not None else 0

        # Determine the group and set file paths for the database and info
        Group = convert_PID_to_PIDgroup(pid, self.CRFTC_RANGE_SIZE)
        bucket_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        db_path = os.path.join(bucket_directory, Group + '.db')
        info_path = os.path.join(bucket_directory, 'info.db')

        # Create the bucket directory if it doesn't exist
        os.makedirs(bucket_directory, exist_ok=True)
        
        # check whether the case is in info_path or not.
        conn = sqlite3.connect(info_path)
        info_query = f'''SELECT * FROM "Table" WHERE PID = ? AND PredDT = ?'''
        try:
            df_case_info = pd.read_sql_query(info_query, conn, params=(pid, str(preddt)))
        except:
            df_case_info = pd.DataFrame(columns = ['PID', 'PredDT', 'nrow'])
        
        # len(df_case_info) > 0, which means we have it already.
        if len(df_case_info) > 0: conn.close(); return None 
        
        # len(df_case_info) == 0, here we need to update the database.
        # 1. update info_path
        df_new = pd.DataFrame([{'PID': pid, 'PredDT': str(preddt), 'nrow': nrow}])
        df_new.to_sql("Table", conn, if_exists='append', index=False)

        cursor = conn.cursor()
        create_index_query = f'''CREATE INDEX IF NOT EXISTS idx_pid_preddt ON "Table" (PID, PredDT);'''
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
        
        CRFTCValue.to_sql("Table", conn, if_exists='append', index=False)

        cursor = conn.cursor()
        create_index_query = f'''CREATE INDEX IF NOT EXISTS idx_pid_preddt ON "Table" (PID, PredDT);'''
        cursor.execute(create_index_query)
        conn.commit()
        conn.close()

    
    def __getitem__(self, index):
        Case = None
        
        # if self.use_cache: 
        #     Case = self.get_cache_case(index)
        # if Case is not None: 
        #     return Case
    
        if self.use_db: 
            Case = self.get_db_case(index)
        if Case is not None: 
            # self.add_to_cache(Case)
            return Case
    
        Case = self.execute_case(index)
        # self.add_to_cache(Case)
        if self.use_db:
            self.add_to_db(Case)
        return Case
    
    def __len__(self):
        return len(self.df_PDT_all)
    