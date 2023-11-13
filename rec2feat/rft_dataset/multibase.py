import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from functools import reduce
from .ckpd import CkpdDataset
from .ckpdrec import CkpdRecDataset
from .ckpdrecflt import CkpdRecFltDataset
from .crftkn import CkpdRecFltTknDataset
from .crftkncmp import CkpdRecFltTknCmpDataset

def load_CRFTC_Dataset(df_PDT_all, CONFIG_PDT, CONFIG_CRFTC, CaseDB_Path, 
                       RANGE_SIZE, CACHE_NUM, UNK_TOKEN, UTILS_Flt, UTILS_CMP, CASE_CACHE_SIZE):
    CONFIG_Ckpd = CONFIG_CRFTC['CONFIG_Ckpd']
    CDataset = CkpdDataset(df_PDT_all, CONFIG_Ckpd, CaseDB_Path, RANGE_SIZE, CASE_CACHE_SIZE)
    if 'CONFIG_RecDB' not in CONFIG_CRFTC: return CDataset

    CONFIG_RecDB = CONFIG_CRFTC['CONFIG_RecDB']
    CRDataset = CkpdRecDataset(CDataset, CONFIG_PDT, CONFIG_RecDB, CaseDB_Path, RANGE_SIZE, CACHE_NUM, CASE_CACHE_SIZE)
    if 'CONFIG_Flt' not in CONFIG_CRFTC: return CRDataset

    CONFIG_Flt = CONFIG_CRFTC['CONFIG_Flt']
    CRFDataset = CkpdRecFltDataset(CRDataset, CONFIG_Flt, UTILS_Flt, CaseDB_Path, RANGE_SIZE, CASE_CACHE_SIZE)
    if 'CONFIG_TknDB' not in CONFIG_CRFTC: return CRFDataset
    
    CONFIG_TknDB = CONFIG_CRFTC['CONFIG_TknDB']
    CRFTDataset = CkpdRecFltTknDataset(CRFDataset, CONFIG_TknDB, CaseDB_Path, UNK_TOKEN, RANGE_SIZE, CASE_CACHE_SIZE)
    if 'CONFIG_CMP' not in CONFIG_CRFTC: return CRFTDataset

    CONFIG_CMP = CONFIG_CRFTC['CONFIG_CMP']
    CRFTCDataset = CkpdRecFltTknCmpDataset(CRFTDataset, CONFIG_CMP, UTILS_CMP, CaseDB_Path, RANGE_SIZE, CASE_CACHE_SIZE)
    return CRFTCDataset


class MultiCRFTCDataset(Dataset):
    def __init__(self, df_PDT_all, 
                 CONFIG_PDT, 
                 CONFIG_CRFTC_List, 
                 CaseDB_Path, **HYPER_DICT):
        
        self.df_PDT_all = df_PDT_all
        self.CONFIG_PDT = CONFIG_PDT
        self.CaseDB_Path = CaseDB_Path
        self.CONFIG_CRFTC_List = CONFIG_CRFTC_List
        
        self.RANGE_SIZE = HYPER_DICT['RANGE_SIZE']
        self.CACHE_NUM = HYPER_DICT['CACHE_NUM']
        self.UTILS_Flt = HYPER_DICT['UTILS_Flt']
        
        self.NameCRFTC_To_Dataset = {}
        for CONFIG_CRFTC in CONFIG_CRFTC_List:
            Dataset = load_CRFTC_Dataset(df_PDT_all, CONFIG_PDT, CONFIG_CRFTC, CaseDB_Path, **HYPER_DICT)
            NameCRFTC = Dataset.NameCRFTC
            self.NameCRFTC_To_Dataset[NameCRFTC] = Dataset
    
    def get_NameCRFTC_to_Case(self, index):
        NameCRFTC_to_Case = {}
        for NameCRFTC, Dataset in self.NameCRFTC_To_Dataset.items():
            Case = Dataset[index] # <------ check whether the data is in db
            NameCRFTC_to_Case[NameCRFTC] = Case
        return NameCRFTC_to_Case
    
    def __getitem__(self, index):
        d = {}
        NameCRFTC_to_Case = self.get_NameCRFTC_to_Case(index)
        for NameCRFTC, Case in NameCRFTC_to_Case.items():
            d['PID'] = Case['PID']
            d['PredDT'] = Case['PredDT']
            d[NameCRFTC] = Case[NameCRFTC]
        return pd.Series(d)
    
    def __len__(self):
        return len(self.df_PDT_all)