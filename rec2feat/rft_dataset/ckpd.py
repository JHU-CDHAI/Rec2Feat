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
        self.df_PDT_all = df_PDT_all
        self.CONFIG_Ckpd = CONFIG_Ckpd
        self.CkpdName = self.get_CkpdName()
        
        # must have
        self.RANGE_SIZE = RANGE_SIZE
        self.CaseDB_Path = CaseDB_Path
        
        # LastDataset
        self.LastDataset = DT2Dataset(self.df_PDT_all)
        self.NameCRFTC = self.CkpdName
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        # db part
        self.create_db_and_tables_done_list = []
        
    def get_CkpdName(self):
        CkpdName = self.CONFIG_Ckpd['CkpdName']
        return CkpdName
        
    def excecute_case(self, index):
        Case = self.LastDataset[index].copy()
        # ----------------
        Case = process_CONFIG_Ckpd_of_PDTInfo(Case, self.CONFIG_Ckpd)
        # -----------------
        return Case
    
    # def __getitem__(self, index):
    #     # overwrite the old __getitem__. 
    #     Case = self.excecute_case(index)
    #     return Case
    