from .base import CRFTC_Base
from ..utils.ckpd import process_CONFIG_Ckpd_of_PDTInfo
from collections import OrderedDict

class DT2Dataset:
    def __init__(self, df):
        self.df = df
    def __getitem__(self, index):
        return self.df.iloc[index]
    def __len__(self):
        return len(self.df)

class CkpdDataset(CRFTC_Base):
    def __init__(self, df_PDT_all, CONFIG_Ckpd, 
                 CaseDB_Path, CRFTC_RANGE_SIZE, CASE_CACHE_SIZE, 
                 use_db, use_cache = False, **kwargs):
        self.CONFIG_Ckpd = CONFIG_Ckpd
        self.CkpdName = self.get_CkpdName()
        
        # must have
        self.CRFTC_RANGE_SIZE = CRFTC_RANGE_SIZE
        self.CaseDB_Path = CaseDB_Path
        self.use_db = use_db
        self.use_cache = use_cache
        
        # LastDataset
        self.df_PDT_all = df_PDT_all
        self.LastDataset = DT2Dataset(self.df_PDT_all)
        self.NameCRFTC = self.CkpdName
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
    def get_CkpdName(self):
        CkpdName = self.CONFIG_Ckpd['CkpdName']
        return CkpdName
        
    def execute_case(self, index):
        # pay attention here, when doing execute, we use LastDataset. 
        Case = self.LastDataset[index].copy()
        Case = process_CONFIG_Ckpd_of_PDTInfo(Case, self.CONFIG_Ckpd)
        return Case