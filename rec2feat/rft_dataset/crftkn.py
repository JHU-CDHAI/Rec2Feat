from torch.utils.data import Dataset
from ..utils.vocab import build_SynFldVocab
from ..utils.crftkn import get_df_TknDB_of_PDTInfo, process_CONFIG_TknDB_of_PDTInfoCRF
from .base import CRFTC_Base
from collections import OrderedDict

class CkpdRecFltTknDataset(CRFTC_Base):

    df_TknDB_Store = {}
    def __init__(self, CRFDataset, CONFIG_TknDB, 
                 CaseDB_Path, CRFTC_RANGE_SIZE, CASE_CACHE_SIZE,
                 GROUP_RANGE_SIZE, GROUP_CACHE_NUM, UNK_TOKEN,
                 use_db, use_cache = False,  **kwargs):
        
        self.CRFDataset = CRFDataset
        self.CONFIG_PDT = CRFDataset.CRDataset.CONFIG_PDT
        self.CONFIG_TknDB = CONFIG_TknDB
        
        self.UNK_TOKEN = UNK_TOKEN # TODO: update to special token list in the future
        
        
        self.GROUP_CACHE_NUM = GROUP_CACHE_NUM
        self.GROUP_RANGE_SIZE = GROUP_RANGE_SIZE
        
        self.SynFldVocab = self.get_SynFldVocab()
        self.SynFldGrn, self.CkpdRecFltTkn = self.get_CkpdRecFltTkn_Name()
        
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.CRFTC_RANGE_SIZE = CRFTC_RANGE_SIZE
        self.use_db = use_db
        self.use_cache = use_cache
        
        # last dataset
        self.df_PDT_all = self.CRFDataset.df_PDT_all
        self.LastDataset = self.CRFDataset
        self.NameCRFTC = self.CkpdRecFltTkn
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        
    def get_CkpdRecFltTkn_Name(self):
        CkpdRecFlt = self.CRFDataset.CkpdRecFlt
        Ckpd, RecName, FilterName = CkpdRecFlt.split('.')
        GrnName = self.CONFIG_TknDB['GrnName']
        SynFldGrn = f'Rof{RecName}-{GrnName}'
        CkpdRecFltTkn = CkpdRecFlt + '.' + SynFldGrn
        return SynFldGrn, CkpdRecFltTkn
        
    def get_SynFldVocab(self):
        fldgrnv_folder = self.CONFIG_TknDB['fldgrnv_folder']
        RecFldGrn_List = self.CONFIG_TknDB['RecFldGrn_List'] 
        RDTCal_rfg = self.CONFIG_TknDB['RDTCal_rfg'] 
        SynFldVocab = build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, self.UNK_TOKEN) 
        return SynFldVocab
        
    def __len__(self):
        return len(self.CRFDataset)
    
    def execute_case(self, index):
        
        Case_CRF = self.LastDataset[index]
        
        # (1) intake and update self.df_TknDB_Store.
        SynFldGrn, df_TknDB, self.df_TknDB_Store = get_df_TknDB_of_PDTInfo(Case_CRF, 
                                                                           self.SynFldGrn, 
                                                                           self.CONFIG_PDT, 
                                                                           self.CONFIG_TknDB, 
                                                                           self.df_TknDB_Store, 
                                                                           self.GROUP_RANGE_SIZE)


        # (2) load PDTInfo_CkpdRecFlt
        Case_CRFT = process_CONFIG_TknDB_of_PDTInfoCRF(Case_CRF, 
                                                       self.CkpdRecFltTkn, 
                                                       self.CONFIG_TknDB, 
                                                       self.df_TknDB_Store, 
                                                       self.UNK_TOKEN)

        
        # (3) adjust size of df_RecDB
        Group_List = list(df_TknDB['Group'].unique())
        if len(Group_List) >= self.GROUP_CACHE_NUM:
            first_Group = df_TknDB.iloc[0]['Group']
            last_Group = df_TknDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_TknDB = df_TknDB[df_TknDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_TknDB_Store[SynFldGrn] = df_TknDB
            
        return Case_CRFT
    
    