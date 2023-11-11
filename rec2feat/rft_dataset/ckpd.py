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
    def __init__(self, df_PDT_all, CONFIG_Ckpd, CaseDB_Path, RANGE_SIZE, CASE_CACHE_SIZE):
        self.CONFIG_Ckpd = CONFIG_Ckpd
        self.CkpdName = self.get_CkpdName()
        
        # must have
        self.RANGE_SIZE = RANGE_SIZE
        self.CaseDB_Path = CaseDB_Path
        
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
        # ----------------
        Case = process_CONFIG_Ckpd_of_PDTInfo(Case, self.CONFIG_Ckpd)
        # -----------------
        return Case
    
    
    def __getitem__(self, index):
        # print('conduct get_cache_case')
        # Case = self.get_cache_case(index)
        # if Case is not None: 
        #     print('conduct get_cache_case success')
        #     return Case
    
        # Case = self.get_bucket_case(index)
        # if Case is not None: 
        #     self.add_to_cache(Case)
        #     return Case
        
        # print('conduct get_db_case')
        Case = self.get_db_case(index)
        if Case is not None: 
            # self.add_to_cache(Case)
            # print('conduct add_to_cache')
            return Case
    
        # print('conduct execute_case')
        Case = self.execute_case(index)
        # execute done: add cache
        # print('conduct add_to_cache')
        # self.add_to_cache(Case)
        # execute done: add db
        # print('conduct add_to_db')
        self.add_to_db(Case)
        return Case