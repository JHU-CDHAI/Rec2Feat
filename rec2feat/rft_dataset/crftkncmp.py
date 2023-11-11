from torch.utils.data import Dataset
from ..utils.vocab import get_update_SynFldVocab
from ..utils.crftkncmp import get_CkpdRecFltTknCmp_Name, process_CONFIG_CMP_of_PDTInfoCRFT
from .base import CRFTC_Base
from collections import OrderedDict

class CkpdRecFltTknCmpDataset(CRFTC_Base):
    
    def __init__(self, CRFTDataset, CONFIG_CMP, UTILS_CMP, CaseDB_Path, RANGE_SIZE, CASE_CACHE_SIZE):
        self.CRFTDataset = CRFTDataset
        self.CkpdRecFltTkn = self.CRFTDataset.CkpdRecFltTkn
        
        self.CONFIG_CMP = CONFIG_CMP
        self.CompressArgs = CONFIG_CMP['CompressArgs']
        self.prefix_layer_cols = CONFIG_CMP['prefix_layer_cols']
        self.focal_layer_cols = CONFIG_CMP['focal_layer_cols']
        
        UTILS_CMP = UTILS_CMP.copy()
        for k, v in CONFIG_CMP['custimized_cmpfn'].items(): 
            UTILS_CMP['method_to_fn'][k] = v
        self.UTILS_CMP = UTILS_CMP
        self.method_to_fn = self.UTILS_CMP['method_to_fn']
        
        self.CkpdRecFltTknCmp = self.get_CkpdRecFltTknCmp_Name()
        self.SynFldVocabNew = self.get_SynFldVocabNew()
        
        # must have
        self.CaseDB_Path = CaseDB_Path
        self.RANGE_SIZE = RANGE_SIZE
        
        # last dataset
        self.df_PDT_all = self.CRFTDataset.df_PDT_all
        self.LastDataset = self.CRFTDataset
        self.NameCRFTC = self.CkpdRecFltTknCmp
        
        # cache part
        self.cache_size = CASE_CACHE_SIZE
        self.Cache_Store = OrderedDict()
        
        
    def get_CkpdRecFltTknCmp_Name(self):
        CkpdRecFltTkn = self.CRFTDataset.CkpdRecFltTkn
        CkpdRecFltTknCmp = get_CkpdRecFltTknCmp_Name(CkpdRecFltTkn, self.CompressArgs, 
                                                     self.prefix_layer_cols, self.focal_layer_cols)
        return CkpdRecFltTknCmp
        
    def get_SynFldVocabNew(self):
        SynFldVocab = self.CRFTDataset.SynFldVocab
        SynFldVocabNew = get_update_SynFldVocab(SynFldVocab, self.CompressArgs)
        return SynFldVocabNew
    
    def execute_case(self, index):
        
        Case_CRFT = self.CRFTDataset[index]
        Case_CRFTC = process_CONFIG_CMP_of_PDTInfoCRFT(Case_CRFT, 
                                                       self.CkpdRecFltTkn, 
                                                       self.CONFIG_CMP, 
                                                       self.UTILS_CMP, 
                                                       self.SynFldVocabNew)
        # print('conduct execute_case')
        return Case_CRFTC
    
    
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