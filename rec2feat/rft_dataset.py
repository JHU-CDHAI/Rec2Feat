import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
from .rft_fn import get_df_RecDB_of_PDTInfo, process_CONFIG_CkpdRecNameFlt_of_PDTInfo
from .rft_fn import get_df_TknDB_of_PDTInfo, process_CONFIG_TknGrn_of_PDTInfoCRF
from .rft_fn import process_CONFIG_CMP_of_PDTInfoCRFT
from .utils_vocab import build_SynFldVocab, get_update_SynFldVocab_and_GrnSeqName

CACHE_NUM = 10

def collate_fn_for_df_PDT(batch_input):
    ##############
    # inputs: you can check the following inputs in the above cells.
    # (1): relational_list
    # (2): new_full_recfldgrn
    # (3): suffix
    ##############
    df_PDT = pd.DataFrame([i for i in batch_input]).reset_index(drop = True)
    return df_PDT


class CkpdRecFltDataset(Dataset):
    
    def __init__(self, df_PDT_all, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                 UTILS_Flt = None, RANGE_SIZE = None, CACHE_NUM = CACHE_NUM):
        self.df_PDT_all = df_PDT_all
        self.CONFIG_PDT = CONFIG_PDT
        self.CONFIG_CkpdRecNameFlt = CONFIG_CkpdRecNameFlt
        self.UTILS_Flt = UTILS_Flt
        self.RANGE_SIZE = RANGE_SIZE
        self.CACHE_NUM = CACHE_NUM
        self.CkpdRecFlt = self.get_CkpdRecFlt_Name()
        
        # this might not be shared across different CkpdRecFltDataset instances.
        self.df_RecDB_Store = {}
        

    def get_CkpdRecFlt_Name(self):
        CkpdName = self.CONFIG_CkpdRecNameFlt['CONFIG_Ckpd']['CkpdName'] # 'Bf1D'
        RecName =  self.CONFIG_CkpdRecNameFlt['CONFIG_RecDB']['RecName']
        FilterName = self.CONFIG_CkpdRecNameFlt['CONFIG_Flt']['FilterName'] # 'Whl'
        CkpdRec = CkpdName + '.' + RecName
        CkpdRecFlt = CkpdRec + '.' + FilterName
        return CkpdRecFlt
    
        
    def __len__(self):
        return len(self.df_PDT_all)
    
    def __getitem__(self, index):
        # df_PDT = self.df_PDT_all[index: index + 1].reset_index(drop = True)
        PDTInfo = self.df_PDT_all.iloc[index]
        CONFIG_PDT = self.CONFIG_PDT
        CONFIG_CkpdRecNameFlt = self.CONFIG_CkpdRecNameFlt
        UTILS_Flt = self.UTILS_Flt
        
        # (1) intake and update self.df_RecDB_Store.
        RecName, df_RecDB, self.df_RecDB_Store = get_df_RecDB_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                                                                         self.df_RecDB_Store, self.RANGE_SIZE)
        
        
        # print(len(df_RecDB), '<---- df_RecDB len')
        # (2) load PDTInfo_CkpdRecFlt
        PDTInfo_CkpdRecFlt = process_CONFIG_CkpdRecNameFlt_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, 
                                                                      self.df_RecDB_Store, self.UTILS_Flt)
        
        # (3) adjust size of df_RecDB
        Group_List = list(df_RecDB['Group'].unique())
        if len(Group_List) >= self.CACHE_NUM:
            first_Group = df_RecDB.iloc[0]['Group']
            last_Group = df_RecDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_RecDB = df_RecDB[df_RecDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_RecDB_Store[RecName] = df_RecDB
            
        return PDTInfo_CkpdRecFlt
    


class CkpdRecFltTknDataset(CkpdRecFltDataset):
    
    def __init__(self, CRFDataset, CONFIG_TknDB, UNK_TOKEN):
        self.CRFDataset = CRFDataset
        self.CONFIG_PDT = CRFDataset.CONFIG_PDT
        self.CONFIG_CkpdRecNameFlt = CRFDataset.CONFIG_CkpdRecNameFlt
        self.CONFIG_TknDB = CONFIG_TknDB
        
        self.RANGE_SIZE = CRFDataset.RANGE_SIZE
        self.CACHE_NUM = CRFDataset.CACHE_NUM
        self.UNK_TOKEN = UNK_TOKEN # TODO: update to special token list in the future
        
        self.SynFldVocab = self.get_SynFldVocab()
        self.SynFldGrn, self.CkpdRecFltTkn = self.get_CkpdRecFltTkn_Name()
        
        # this might not be shared across different CkpdRecFltDataset instances.
        self.df_TknDB_Store = {}
        
    def get_CkpdRecFltTkn_Name(self):
        CkpdRecFlt = self.CRFDataset.CkpdRecFlt
        Ckpd, RecName, FilterName = CkpdRecFlt.split('.')
        GrnName = self.CONFIG_TknDB['GrnName']
        SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
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
    
    def __getitem__(self, index):
        
        PDTInfo_CRF = self.CRFDataset[index]
        
        # (1) intake and update self.df_TknDB_Store.
        SynFldGrn, df_TknDB, self.df_TknDB_Store = get_df_TknDB_of_PDTInfo(PDTInfo_CRF, 
                                                                          self.CONFIG_PDT, 
                                                                          self.CONFIG_TknDB, 
                                                                          self.df_TknDB_Store, 
                                                                          self.RANGE_SIZE)


        # (2) load PDTInfo_CkpdRecFlt
        
        PDTInfo_CRFT = process_CONFIG_TknGrn_of_PDTInfoCRF(PDTInfo_CRF, 
                                                           self.CONFIG_PDT, 
                                                           self.CONFIG_TknDB, 
                                                           self.df_TknDB_Store)
    
        # (3) adjust size of df_RecDB
        Group_List = list(df_TknDB['Group'].unique())
        if len(Group_List) >= self.CACHE_NUM:
            first_Group = df_TknDB.iloc[0]['Group']
            last_Group = df_TknDB.iloc[-1]['Group']
            Groups_to_keep = [i for i in Group_List if i not in [first_Group, last_Group]][1:] + [last_Group]
            df_TknDB = df_TknDB[df_TknDB['Group'].isin(Groups_to_keep)].reset_index(drop = True)
            # df_RecDB = df_RecDB.sample(int(self.CACHE_NUM / 6 * 5))
            self.df_TknDB_Store[SynFldGrn] = df_TknDB
            
        return PDTInfo_CRFT




class CkpdRecFltTknCmpDataset(Dataset):
    
    def __init__(self, CRFTDataset, CONFIG_CMP, UTILS_CMP, UNK_TOKEN):
        self.CRFTDataset = CRFTDataset
        self.CONFIG_PDT = CRFTDataset.CONFIG_PDT
        self.CONFIG_CkpdRecNameFlt = CRFTDataset.CONFIG_CkpdRecNameFlt
        self.CONFIG_TknDB = CRFTDataset.CONFIG_TknDB
        self.CONFIG_CMP = CONFIG_CMP
        
        self.UTILS_CMP = UTILS_CMP
        
        self.RANGE_SIZE = CRFTDataset.RANGE_SIZE
        self.CACHE_NUM = CRFTDataset.CACHE_NUM
        self.UNK_TOKEN = UNK_TOKEN # TODO: update to special token list in the future
        
        self.SynFldVocab = self.get_SynFldVocab()
        self.SynFldVocabNew, self.CkpdRecFltGrnCmp, self.CkpdRecFltGrnCmpFeat = self.get_SynFldVocabNew_and_FeatNames()
        
    def get_SynFldVocab(self):
        fldgrnv_folder = self.CONFIG_TknDB['fldgrnv_folder']
        RecFldGrn_List = self.CONFIG_TknDB['RecFldGrn_List'] 
        RDTCal_rfg = self.CONFIG_TknDB['RDTCal_rfg'] 
        SynFldVocab = build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, self.UNK_TOKEN) 
        return SynFldVocab
    
    def get_SynFldVocabNew_and_FeatNames(self):
        PDTName = self.CONFIG_PDT['PDTName']
        
        CkpdName, RecName, FilterName, SynFldGrn = self.CRFTDataset.CkpdRecFltTkn.split('.')
        
        SynFldVocab = self.SynFldVocab
        CompressArgs = self.CONFIG_CMP['CompressArgs']
        prefix_layer_cols = self.CONFIG_CMP['prefix_layer_cols']
        focal_layer_cols = self.CONFIG_CMP['focal_layer_cols']
        
        SynFldVocabNew, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat = \
            get_update_SynFldVocab_and_GrnSeqName(PDTName, CkpdName, RecName, FilterName, SynFldGrn, 
                                                  SynFldVocab, CompressArgs, prefix_layer_cols, focal_layer_cols)
        
        return SynFldVocabNew, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat

    def __len__(self):
        return len(self.CRFTDataset)
    
    def __getitem__(self, index):
        
        PDTInfo_CRFT = self.CRFTDataset[index]
        PDTInfo_CRFTC = process_CONFIG_CMP_of_PDTInfoCRFT(
                                      PDTInfo_CRFT, self.CONFIG_PDT, self.CONFIG_TknDB, self.CONFIG_CMP, 
                                      self.SynFldVocabNew, self.CkpdRecFltGrnCmp, self.CkpdRecFltGrnCmpFeat, 
                                      UTILS_CMP = self.UTILS_CMP, UNK_TOKEN = self.UNK_TOKEN)
            
        return PDTInfo_CRFTC
