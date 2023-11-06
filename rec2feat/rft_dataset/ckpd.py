from torch.utils.data import Dataset
from ..utils.ckpd import process_CONFIG_Ckpd_of_PDTInfo


class CkpdDataset(Dataset):
    
    def __init__(self, df_PDT_all, CONFIG_Ckpd):
        self.df_PDT_all = df_PDT_all
        self.CONFIG_Ckpd = CONFIG_Ckpd
        self.CkpdName = self.get_CkpdName()
    
    def get_CkpdName(self):
        CkpdName = self.CONFIG_Ckpd['CkpdName']
        return CkpdName
    
    def __len__(self):
        return len(self.df_PDT_all)
    
    def __getitem__(self, index):
        Case = self.df_PDT_all.iloc[index]
        Case_C = process_CONFIG_Ckpd_of_PDTInfo(Case, self.CONFIG_Ckpd)
        return Case_C
    
    