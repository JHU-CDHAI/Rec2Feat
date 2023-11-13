import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import OrderedDict
from ..utils.ckpdrecflt import process_CONFIG_CkpdRecFlt_of_PDTInfo
from .base import CRFTC_Base


class CkpdRecFltDataset(CRFTC_Base):
    # this might not be shared across different CkpdRecFltDataset instances.
    
    def __init__(self, CRDataset, CONFIG_Flt, UTILS_Flt, 
                 CaseDB_Path, CRFTC_RANGE_SIZE, CASE_CACHE_SIZE, 
                 use_db, use_cache = False, **kwargs):
        
        self.CRDataset = CRDataset
        self.CONFIG_Flt = CONFIG_Flt
        self.CkpdRecFlt = self.get_CkpdRecFlt_Name()
        
        UTILS_Flt = UTILS_Flt.copy()
        CONFIG = CONFIG_Flt.get('Customized_FilterName2Share', {})
        for k, v in CONFIG.items(): UTILS_Flt['FilterName2Share'][k] = v
        CONFIG = CONFIG_Flt.get('Customized_RecName2ThresConfig', {})
        for k, v in CONFIG.items(): UTILS_Flt['RecName2ThresConfig'][k] = v

        self.UTILS_Flt = UTILS_Flt
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.CRFTC_RANGE_SIZE = CRFTC_RANGE_SIZE
        self.use_db = use_db
        self.use_cache = use_cache
        
        # last dataset
        self.df_PDT_all = self.CRDataset.df_PDT_all
        self.LastDataset = self.CRDataset
        self.NameCRFTC = self.CkpdRecFlt
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
    def get_CkpdRecFlt_Name(self):
        CkpdRec = self.CRDataset.CkpdRec
        FilterName = self.CONFIG_Flt['FilterName']
        CkpdRecFlt = CkpdRec + '.' + FilterName
        return CkpdRecFlt
    
    def execute_case(self, index):
        Case_CR = self.LastDataset[index]
        CkpdRec = self.LastDataset.CkpdRec
        Case_CRF = process_CONFIG_CkpdRecFlt_of_PDTInfo(Case_CR, CkpdRec, self.CONFIG_Flt, self.UTILS_Flt)
        return Case_CRF
    