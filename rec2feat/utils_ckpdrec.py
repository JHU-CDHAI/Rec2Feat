import pandas as pd
from recfldgrn.loadtools import get_df_bucket_from_settings



def get_info_from_settings(Group, PID_List, RecChain_ARGS, RecInfo_ARGS):
    bucket_file = Group + '.p'
    df = get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS)
    # print('get_info_from_settings', 'PID_List', PID_List)
    if type(PID_List) == list:
        df = df[df['PID'].isin(PID_List)].reset_index(drop = True)
    return df


def get_Ckpd_from_PredDT(PredDT, DistStartToPredDT, DistEndToPredDT, TimeUnit, **kwargs):
    assert TimeUnit in ['H', 'D', 'min']
    DT_s = PredDT + pd.to_timedelta(DistStartToPredDT, unit = TimeUnit)
    DT_e = PredDT + pd.to_timedelta(DistEndToPredDT, unit = TimeUnit)
    return {'DT_s': DT_s, 'DT_e':DT_e}


def get_CkpdName_RecTable(PDTInfo, CkpdName, DBInfo, RecNameTpt, cols = None):
    CkptPeriod = PDTInfo[CkpdName] 

    if RecNameTpt not in list(DBInfo.keys()): return None
    df = DBInfo[RecNameTpt]
    if type(df) != type(pd.DataFrame()) or len(df) == 0: return None
    df = df.copy()
    index_s = df['DT'] >= CkptPeriod['DT_s']
    index_e = df['DT'] <  CkptPeriod['DT_e'] # not <=
    df = df.loc[index_s & index_e].reset_index(drop = True)
    if len(df) == 0: return None
    RecName = RecNameTpt.split('_')[0]
    # cols = [RecName + 'ID', 'DT']
    df['PredDT'] = PDTInfo['PredDT']
    if type(cols) != list: cols = list(df.columns)
    cols = ['PID', 'PredDT'] + [i for i in cols if i not in ['PID', 'PredDT']]
    df = df[cols]# .copy()
    return df


