import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
from ..utils.ckpdrec import get_df_RecDB_of_PDTInfo, process_CONFIG_CkpdRec_of_PDTInfo
from collections import OrderedDict
from .base import CRFTC_Base


class CkpdRecDataset(CRFTC_Base):
    # this might not be shared across different CkpdRecFltDataset instances.
    df_RecDB_Store = {}
    
    def __init__(self, CDataset, CONFIG_PDT, CONFIG_RecDB, CaseDB_Path, 
                 RANGE_SIZE = None, CACHE_NUM = None, CASE_CACHE_SIZE = None):
        
        
        self.CDataset = CDataset
        self.CONFIG_PDT = CONFIG_PDT
        self.CONFIG_RecDB = CONFIG_RecDB
        self.CACHE_NUM = CACHE_NUM
        self.CkpdName = CDataset.CkpdName
        self.CkpdRec = self.get_CkpdRec_Name()
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.RANGE_SIZE = RANGE_SIZE
        
        # last dataset
        self.LastDataset = self.CDataset
        self.NameCRFTC = self.CkpdRec
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        # db part
        self.create_db_and_tables_done_list = []
        
        
    def get_CkpdRec_Name(self):
        CkpdName = self.CDataset.CkpdName
        RecName =  self.CONFIG_RecDB['RecName']
        CkpdRec = CkpdName + '.' + RecName
        return CkpdRec
    
    def excecute_case(self, index):
        Case_C = self.CDataset[index]
        
        # (1) update self.df_RecDB_Store and get df_RecDB
        RecName, df_RecDB, self.df_RecDB_Store = get_df_RecDB_of_PDTInfo(
                                                        Case_C, 
                                                        self.CONFIG_PDT, 
                                                        self.CONFIG_RecDB, 
                                                        self.df_RecDB_Store, 
                                                        self.RANGE_SIZE)
        
        # (2) load PDTInfo_CkpdRecFlt
        Case_CR = process_CONFIG_CkpdRec_of_PDTInfo(Case_C, 
                                                           self.CkpdName, 
                                                           self.CONFIG_RecDB, 
                                                           self.df_RecDB_Store)
        
        # (3) adjust size of df_RecDB
        Group_List = list(df_RecDB['Group'].unique())
        if len(Group_List) >= self.CACHE_NUM:
            first_Group = df_RecDB.iloc[0]['Group']
            last_Group = df_RecDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_RecDB = df_RecDB[df_RecDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_RecDB_Store[RecName] = df_RecDB
            
        return Case_CR
