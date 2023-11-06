from torch.utils.data import Dataset
from ..utils.vocab import get_update_SynFldVocab
from ..utils.crftkncmp import get_CkpdRecFltTknCmp_Name, process_CONFIG_CMP_of_PDTInfoCRFT

class CkpdRecFltTknCmpDataset(Dataset):
    
    def __init__(self, CRFTDataset, CONFIG_CMP, UTILS_CMP):
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
        
    def get_CkpdRecFltTknCmp_Name(self):
        CkpdRecFltTkn = self.CRFTDataset.CkpdRecFltTkn
        CkpdRecFltTknCmp = get_CkpdRecFltTknCmp_Name(CkpdRecFltTkn, self.CompressArgs, 
                                                     self.prefix_layer_cols, self.focal_layer_cols)
        return CkpdRecFltTknCmp
        
    def get_SynFldVocabNew(self):
        SynFldVocab = self.CRFTDataset.SynFldVocab
        SynFldVocabNew = get_update_SynFldVocab(SynFldVocab, self.CompressArgs)
        return SynFldVocabNew
        
    def __len__(self):
        return len(self.CRFTDataset)
    
    def __getitem__(self, index):
        
        Case_CRFT = self.CRFTDataset[index]
        Case_CRFTC = process_CONFIG_CMP_of_PDTInfoCRFT(Case_CRFT, 
                                                       self.CkpdRecFltTkn, 
                                                       self.CONFIG_CMP, 
                                                       self.UTILS_CMP, 
                                                       self.SynFldVocabNew)
        return Case_CRFTC
    