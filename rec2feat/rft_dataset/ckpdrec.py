from collections import OrderedDict
from ..utils.ckpdrec import get_df_RecDB_of_PDTInfo, process_CONFIG_CkpdRec_of_PDTInfo
from .base import CRFTC_Base

class CkpdRecDataset(CRFTC_Base):
    # this might not be shared across different CkpdRecFltDataset instances.
    df_RecDB_Store = {}
    
    def __init__(self, CDataset, CONFIG_PDT, CONFIG_RecDB, 
                 CaseDB_Path, CRFTC_RANGE_SIZE, CASE_CACHE_SIZE, 
                 GROUP_RANGE_SIZE, GROUP_CACHE_NUM, 
                 use_db, use_cache = False, **kwargs):
        
        
        self.CDataset = CDataset
        self.CONFIG_PDT = CONFIG_PDT
        self.CONFIG_RecDB = CONFIG_RecDB
        
        self.CkpdName = CDataset.CkpdName
        self.CkpdRec = self.get_CkpdRec_Name()
        
        self.GROUP_CACHE_NUM = GROUP_CACHE_NUM
        self.GROUP_RANGE_SIZE = GROUP_RANGE_SIZE
        
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.CRFTC_RANGE_SIZE = CRFTC_RANGE_SIZE
        self.use_db = use_db
        self.use_cache = use_cache
        
        # last dataset
        self.df_PDT_all = self.CDataset.df_PDT_all
        self.LastDataset = self.CDataset
        self.NameCRFTC = self.CkpdRec
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        
    def get_CkpdRec_Name(self):
        CkpdName = self.CDataset.CkpdName
        RecName =  self.CONFIG_RecDB['RecName']
        CkpdRec = CkpdName + '.' + RecName
        return CkpdRec
    
    def execute_case(self, index):
        Case_C = self.LastDataset[index]
        
        # (1) update self.df_RecDB_Store and get df_RecDB
        RecName, df_RecDB, self.df_RecDB_Store = get_df_RecDB_of_PDTInfo(
                                                        Case_C, 
                                                        self.CONFIG_PDT, 
                                                        self.CONFIG_RecDB, 
                                                        self.df_RecDB_Store, 
                                                        self.GROUP_RANGE_SIZE)
        
        # (2) load PDTInfo_CkpdRecFlt
        Case_CR = process_CONFIG_CkpdRec_of_PDTInfo(Case_C, 
                                                   self.CkpdName, 
                                                   self.CONFIG_RecDB, 
                                                   self.df_RecDB_Store)
        
        # (3) adjust size of df_RecDB
        Group_List = list(df_RecDB['Group'].unique())
        if len(Group_List) >= self.GROUP_CACHE_NUM:
            first_Group = df_RecDB.iloc[0]['Group']
            last_Group = df_RecDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_RecDB = df_RecDB[df_RecDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_RecDB_Store[RecName] = df_RecDB
            
        return Case_CR
