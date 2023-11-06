import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from ..utils.ckpdrecflt import process_CONFIG_CkpdRecFlt_of_PDTInfo


class CkpdRecFltDataset(Dataset):
    # this might not be shared across different CkpdRecFltDataset instances.
    
    def __init__(self, CRDataset, CONFIG_Flt, UTILS_Flt):
        self.CRDataset = CRDataset
        self.CONFIG_Flt = CONFIG_Flt
        self.CkpdRecFlt = self.get_CkpdRecFlt_Name()
        
        UTILS_Flt = UTILS_Flt.copy()
        CONFIG = CONFIG_Flt['Customized_FilterName2Share']
        for k, v in CONFIG.items():UTILS_Flt['FilterName2Share'][k] = v
        CONFIG = CONFIG_Flt['Customized_RecName2ThresConfig']
        for k, v in CONFIG.items():UTILS_Flt['RecName2ThresConfig'][k] = v
        self.UTILS_Flt = UTILS_Flt

        
    def get_CkpdRecFlt_Name(self):
        CkpdRec = self.CRDataset.CkpdRec
        FilterName = self.CONFIG_Flt['FilterName']
        CkpdRecFlt = CkpdRec + '.' + FilterName
        return CkpdRecFlt
    
        
    def __len__(self):
        return len(self.CRDataset)
    
    def __getitem__(self, index):
        Case_CR = self.CRDataset[index]
        CkpdRec = self.CRDataset.CkpdRec
        Case_CRF = process_CONFIG_CkpdRecFlt_of_PDTInfo(Case_CR, CkpdRec, self.CONFIG_Flt, self.UTILS_Flt)
        return Case_CRF