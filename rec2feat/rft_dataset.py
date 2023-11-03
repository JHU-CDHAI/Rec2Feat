import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
from .rft_fn import get_df_RecDB_of_PDTInfo, process_CONFIG_CkpdRecNameFlt_of_PDTInfo


CACHE_NUM = 100


def collate_fn_for_df_PDT_CRF(batch_input):
    ##############
    # inputs: you can check the following inputs in the above cells.
    # (1): relational_list
    # (2): new_full_recfldgrn
    # (3): suffix
    ##############
    df_PDT = pd.concat([i for i in batch_input]).reset_index(drop = True)
    
    return df_PDT





class CkpdRecFltDataset(Dataset):
    
    # df_RecDB: defined by Patient and RecName
    df_RecDB_Store = {}
    # df_TknDB_Store = {}
    
    def __init__(self, df_PDT_all, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                 UTILS_Flt = None, RANGE_SIZE = None, CACHE_NUM = CACHE_NUM):
        self.df_PDT_all = df_PDT_all
        self.CONFIG_PDT = CONFIG_PDT
        self.CONFIG_CkpdRecNameFlt = CONFIG_CkpdRecNameFlt
        self.UTILS_Flt = UTILS_Flt
        self.RANGE_SIZE = RANGE_SIZE
        self.CACHE_NUM = CACHE_NUM
        
    def __len__(self):
        return len(self.df_PDT_all)
    
    def __getitem__(self, index):
        # df_PDT = self.df_PDT_all[index: index + 1].reset_index(drop = True)
        PDTInfo = self.df_PDT_all.iloc[index]
        CONFIG_PDT = self.CONFIG_PDT
        CONFIG_CkpdRecNameFlt = self.CONFIG_CkpdRecNameFlt
        UTILS_Flt = self.UTILS_Flt
        
        # (1) intake and update self.df_RecDB_Store.
        RecName, df_RecDB, self.df_RecDB_Store = get_df_RecDB_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                                                                         self.df_RecDB_Store, self.RANGE_SIZE)
        
        
        # print(len(df_RecDB), '<---- df_RecDB len')
        # (2) load PDTInfo_CkpdRecFlt
        PDTInfo_CkpdRecFlt = process_CONFIG_CkpdRecNameFlt_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                                                                      self.df_RecDB_Store, self.UTILS_Flt)
        
        # (3) adjust size of df_RecDB
        if len(df_RecDB) >= self.CACHE_NUM:
            df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_RecDB_Store[RecName] = df_RecDB
            
        return PDTInfo_CkpdRecFlt