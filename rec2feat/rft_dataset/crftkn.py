from torch.utils.data import Dataset
from ..utils.vocab import build_SynFldVocab
from ..utils.crftkn import get_df_TknDB_of_PDTInfo, process_CONFIG_TknDB_of_PDTInfoCRF
from .base import CRFTC_Base
from collections import OrderedDict

class CkpdRecFltTknDataset(CRFTC_Base):

    df_TknDB_Store = {}

    def __init__(self, CRFDataset, CONFIG_TknDB, CaseDB_Path, UNK_TOKEN, RANGE_SIZE, CASE_CACHE_SIZE):
        self.CRFDataset = CRFDataset
        self.CONFIG_PDT = CRFDataset.CRDataset.CONFIG_PDT
        # self.CONFIG_CkpdRecNameFlt = CRFDataset.CONFIG_CkpdRecNameFlt
        self.CONFIG_TknDB = CONFIG_TknDB
        self.RANGE_SIZE = CRFDataset.CRDataset.RANGE_SIZE
        self.CACHE_NUM = CRFDataset.CRDataset.CACHE_NUM
        self.UNK_TOKEN = UNK_TOKEN # TODO: update to special token list in the future
        
        self.SynFldVocab = self.get_SynFldVocab()
        self.SynFldGrn, self.CkpdRecFltTkn = self.get_CkpdRecFltTkn_Name()
        
        # this might not be shared across different CkpdRecFltDataset instances.
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.RANGE_SIZE = RANGE_SIZE
        
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
    
    
    def execute_case(self, index):
        
        Case_CRF = self.LastDataset[index]
        
        # (1) intake and update self.df_TknDB_Store.
        SynFldGrn, df_TknDB, self.df_TknDB_Store = get_df_TknDB_of_PDTInfo(Case_CRF, 
                                                                           self.SynFldGrn, 
                                                                           self.CONFIG_PDT, 
                                                                           self.CONFIG_TknDB, 
                                                                           self.df_TknDB_Store, 
                                                                           self.RANGE_SIZE)


        # (2) load PDTInfo_CkpdRecFlt
        Case_CRFT = process_CONFIG_TknDB_of_PDTInfoCRF(Case_CRF, 
                                                       self.CkpdRecFltTkn, 
                                                       self.CONFIG_TknDB, 
                                                       self.df_TknDB_Store, 
                                                       self.UNK_TOKEN)

        
    
        # (3) adjust size of df_RecDB
        Group_List = list(df_TknDB['Group'].unique())
        if len(Group_List) >= self.CACHE_NUM:
            first_Group = df_TknDB.iloc[0]['Group']
            last_Group = df_TknDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_TknDB = df_TknDB[df_TknDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_TknDB_Store[SynFldGrn] = df_TknDB
            
        return Case_CRFT
    
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