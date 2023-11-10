import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from functools import reduce
import sqlite3
from collections import OrderedDict
from recfldgrn.datapoint import convert_PID_to_PIDgroup


def get_from_cache(Cache_Store, pid, preddt):
    # Use a tuple of PID and PredDT as the cache key
    cache_key = (pid, preddt)
    return Cache_Store.get(cache_key)

def add_to_cache(Cache_Store, pid, preddt, value, cache_size):
    cache_key = (pid, preddt)
    if cache_key in Cache_Store:
        # Move the key to the end to indicate recent use
        Cache_Store.move_to_end(cache_key)
    elif len(Cache_Store) >= cache_size:
        # Evict the oldest item from the cache (first item)
        Cache_Store.popitem(last=False)
    Cache_Store[cache_key] = value
    return Cache_Store


class CRFTC_Base(Dataset):
    def __init__(self, RANGE_SIZE, CaseDB_Path, CASE_CACHE_SIZE):
        # must have
        self.RANGE_SIZE = RANGE_SIZE
        self.CaseDB_Path = CaseDB_Path
        
        # LastDataset
        self.LastDataset = self.df_PDT_all
        self.NameCRFTC = None
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        # db part
        self.create_db_and_tables_done_list = []
        
        
    def get_cache_case(self, index):
        # get the initial case
        Case = self.LastDataset.iloc[index].copy()
        pid, preddt = str(Case['PID']), str(Case['PredDT'])
        
        # get the name of CRFTC
        NameCRFTC = self.NameCRFTC
        
        # 1. first check whether ValueCRFTC in Cache_Store or not?
        ValueCRFTC = get_from_cache(self.Cache_Store, pid, preddt)
        
        # print(self.Cache_Store)
        # print(self.Cache_Store.get((pid, preddt)))
        
        # if ValueCRFT is not None, return the result. 
        if ValueCRFTC is not None:
            # print('load from self.Cache_Store')
            Case[NameCRFTC] = ValueCRFTC
        else:
            Case = None
        return Case
    
    
    def create_db_and_tables(self, cursor, conn, NameCRFTC):
        # 2.3 Check if the table exists and create it if not
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{NameCRFTC}'")
        
        if cursor.fetchone(): return None
        
        cursor.execute(f'''
            CREATE TABLE {NameCRFTC} (
                PID TEXT,
                PredDT TEXT,
                {NameCRFTC} TEXT,
                PRIMARY KEY (PID, PredDT)
            )
        ''')
        conn.commit()
        self.create_db_and_tables_done_list.append(NameCRFTC)
        
        
    def get_db_case(self, index):
        # get the initial case
        Case = self.LastDataset.iloc[index].copy()
        pid, preddt = str(Case['PID']), str(Case['PredDT'])
        # get the name of CRFTC
        NameCRFTC = self.NameCRFTC
        
        # 2. second check whether ValueCRFTC in Database or not?
        # 2.1 get the db_path
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        db_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        db_path = os.path.join(self.CaseDB_Path, NameCRFTC, Group + '.db')
        os.makedirs(db_directory, exist_ok=True)
    
        # 2.2 Establish a connection to the SQLite DB file
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 2.3 create db and tables
        if NameCRFTC not in self.create_db_and_tables_done_list:
            self.create_db_and_tables(cursor, conn, NameCRFTC)
        
        # 2.4 Prepare the SQL query to check if the entry exists
        query = f"""
        SELECT * FROM {NameCRFTC} 
        WHERE PID = ? AND PredDT = ?
        """
        cursor.execute(query, (pid, preddt))
        result = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        # 2.5 get ValueCRFTC from database
        if result is not None:
            # print(result)
            ValueCRFTC_json = result[-1] # NameCRFTC
            ValueCRFTC = pd.read_json(ValueCRFTC_json, orient='split')
            # get the case
            Case[NameCRFTC] = ValueCRFTC
            # update it to Cache
            self.Cache_Store = add_to_cache(self.Cache_Store, pid, preddt, ValueCRFTC, self.cache_size)
            # print('load from Database')
        else:
            Case = None
        return Case
    
    
    def update_case_to_db_and_cache(self, Case):
        pid, preddt = str(Case['PID']), str(Case['PredDT'])
        NameCRFTC = list(Case.keys())[-1]
        ValueCRFTC = Case[NameCRFTC]
        
        # 3.2 add the ValueCRFTC to the cache
        self.Cache_Store = add_to_cache(self.Cache_Store, pid, preddt, ValueCRFTC, self.cache_size)
            
        # 3.3 write to the database
        ValueCRFTC_json = ValueCRFTC.to_json(orient='split')
        # print(ValueCRFTC_json)
        
        # 2. second check whether ValueCRFTC in Database or not?
        # 2.1 get the db_path
        Group = convert_PID_to_PIDgroup(pid, self.RANGE_SIZE)
        db_directory = os.path.join(self.CaseDB_Path, NameCRFTC)
        db_path = os.path.join(self.CaseDB_Path, NameCRFTC, Group + '.db')
        os.makedirs(db_directory, exist_ok=True)
    
        # 2.2 Establish a connection to the SQLite DB file
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
            
        # Insert the new entry into the database
        insert_query = f"""
        INSERT INTO {NameCRFTC} (PID, PredDT, {NameCRFTC}) 
        VALUES (?, ?, ?)
        """
        # print(insert_query)
        cursor.execute(insert_query, (pid, preddt, ValueCRFTC_json))
        conn.commit()
        conn.close()
        
        
    def __getitem__(self, index):
        Case = self.get_cache_case(index)
        if Case is not None: return Case
    
        Case = self.get_db_case(index)
        if Case is not None: return Case
    
        Case = self.excecute_case(index)
        self.update_case_to_db_and_cache(Case)
        
        return Case
    
    def __len__(self):
        return len(self.LastDataset)