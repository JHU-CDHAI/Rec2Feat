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

def load_CRFTC_Dataset(df_PDT_all, 
                       CONFIG_PDT, 
                       CONFIG_CRFTC, CONFIG_CRFTC_RANGE_SIZE, CONFIG_CRFTC_usedb, 
                       DS_ARGS_Template, 
                       UTILS_Flt,
                       UTILS_CMP):
    
    # Ckpd
    DS_ARGS = DS_ARGS_Template.copy()
    DS_ARGS['use_db'] = CONFIG_CRFTC_usedb['Ckpd'] # False
    DS_ARGS['CRFTC_RANGE_SIZE'] = CONFIG_CRFTC_RANGE_SIZE['Ckpd']
    CONFIG_Ckpd = CONFIG_CRFTC['Ckpd']
    CDataset = CkpdDataset(df_PDT_all, CONFIG_Ckpd, **DS_ARGS)
    if 'RecDB' not in CONFIG_CRFTC: return CDataset

    # RecDB
    DS_ARGS = DS_ARGS_Template.copy()
    DS_ARGS['use_db'] = CONFIG_CRFTC_usedb['RecDB'] 
    DS_ARGS['CRFTC_RANGE_SIZE'] = CONFIG_CRFTC_RANGE_SIZE['RecDB']
    CONFIG_RecDB = CONFIG_CRFTC['RecDB']
    CRDataset = CkpdRecDataset(CDataset, CONFIG_PDT, CONFIG_RecDB, **DS_ARGS)
    if 'Flt' not in CONFIG_CRFTC: return CRDataset

    DS_ARGS = DS_ARGS_Template.copy()
    DS_ARGS['use_db'] = CONFIG_CRFTC_usedb['Flt'] 
    DS_ARGS['CRFTC_RANGE_SIZE'] = CONFIG_CRFTC_RANGE_SIZE['Flt']
    CONFIG_Flt = CONFIG_CRFTC['Flt']
    CRFDataset = CkpdRecFltDataset(CRDataset, CONFIG_Flt, UTILS_Flt, **DS_ARGS)
    if 'TknDB' not in CONFIG_CRFTC: return CRFDataset
    
    DS_ARGS = DS_ARGS_Template.copy()
    DS_ARGS['use_db'] = CONFIG_CRFTC_usedb['TknDB'] 
    DS_ARGS['CRFTC_RANGE_SIZE'] = CONFIG_CRFTC_RANGE_SIZE['TknDB']
    CONFIG_TknDB = CONFIG_CRFTC['TknDB']
    CRFTDataset = CkpdRecFltTknDataset(CRFDataset, CONFIG_TknDB, **DS_ARGS)
    if 'CMP' not in CONFIG_CRFTC: return CRFTDataset

    DS_ARGS = DS_ARGS_Template.copy()
    DS_ARGS['use_db'] = CONFIG_CRFTC_usedb['CMP'] 
    DS_ARGS['CRFTC_RANGE_SIZE'] = CONFIG_CRFTC_RANGE_SIZE['CMP']
    CONFIG_CMP = CONFIG_CRFTC['CMP']
    CRFTCDataset = CkpdRecFltTknCmpDataset(CRFTDataset, CONFIG_CMP, UTILS_CMP, **DS_ARGS)
    return CRFTCDataset


class MultiCRFTCDataset(Dataset):
    def __init__(self, 
                 df_PDT_all, 
                 CONFIG_PDT, 
                 CONFIG_CRFTC_List, 
                 DS_ARGS_Template, 
                 UTILS_Flt, UTILS_CMP):
        
        self.df_PDT_all = df_PDT_all
        self.CONFIG_PDT = CONFIG_PDT
        self.CONFIG_CRFTC_List = CONFIG_CRFTC_List
        self.UTILS_Flt = UTILS_Flt
        self.UTILS_CMP = UTILS_CMP
        
        self.NameCRFTC_To_Dataset = {}
        for crftc_tuple in CONFIG_CRFTC_List:
            CONFIG_CRFTC, CONFIG_CRFTC_RANGE_SIZE, CONFIG_CRFTC_usedb = crftc_tuple
            Dataset = load_CRFTC_Dataset(df_PDT_all, 
                                         CONFIG_PDT,
                                         CONFIG_CRFTC, 
                                         CONFIG_CRFTC_RANGE_SIZE, 
                                         CONFIG_CRFTC_usedb, 
                                         DS_ARGS_Template,
                                         UTILS_Flt,
                                         UTILS_CMP)
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
    