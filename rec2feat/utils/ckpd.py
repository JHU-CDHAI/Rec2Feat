import pandas as pd

def get_Ckpd_from_PredDT(PredDT, DistStartToPredDT, DistEndToPredDT, TimeUnit, **kwargs):
    assert TimeUnit in ['H', 'D', 'min']
    DT_s = PredDT + pd.to_timedelta(DistStartToPredDT, unit = TimeUnit)
    DT_e = PredDT + pd.to_timedelta(DistEndToPredDT, unit = TimeUnit)
    return {'DT_s': DT_s, 'DT_e':DT_e}

def process_CONFIG_Ckpd_of_PDTInfo(Case, CONFIG_Ckpd):
    PDTInfo = Case.copy()
    # -------- 1. CkpdName
    CkpdName = CONFIG_Ckpd['CkpdName'] # 'Bf1D'
    Ckpd_ARGS = {
        'CkpdName':  CONFIG_Ckpd['CkpdName'], 
        'DistStartToPredDT': CONFIG_Ckpd['DistStartToPredDT'],
        'DistEndToPredDT':  CONFIG_Ckpd['DistEndToPredDT'],
        'TimeUnit':  CONFIG_Ckpd['TimeUnit'], 
    }
    PDTInfo[CkpdName] = get_Ckpd_from_PredDT(PDTInfo['PredDT'], **Ckpd_ARGS)
    PDTInfo_Ckpd = PDTInfo
    return PDTInfo_Ckpd

