import pandas as pd
from recfldgrn.loadtools import get_df_bucket_from_settings
from recfldgrn.datapoint import convert_PID_to_PIDgroup


def get_group_info(Group, RecChain_ARGS, RecInfo_ARGS):
    bucket_file = Group + '.p'
    df = get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS)
    df['Group'] = Group
    # if type(PID_List) == list: df = df[df['PID'].isin(PID_List)].reset_index(drop = True)
    return df


def get_df_RecDB_of_PDTInfo(Case_C, CONFIG_PDT, CONFIG_RecDB, df_RecDB_Store, RANGE_SIZE):

    # CONFIG_RecDB = CONFIG_CkpdRecNameFlt['CONFIG_RecDB']
    PDTInfo = Case_C.copy()
    
    # CONFIG_PDT
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    # df_PDT['Group'] = df_PDT['PID'].apply(lambda x: convert_PID_to_PIDgroup(x, RANGE_SIZE))
    Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    
    # CONFIG_RecDB
    rec_folder = CONFIG_RecDB['rec_folder'] 
    RootRec =  CONFIG_RecDB['RootRec']
    tpt_folder =  CONFIG_RecDB['tpt_folder']
    RecName =  CONFIG_RecDB['RecName']
    RecNameTpt =  CONFIG_RecDB['RecNameTpt']
    
    # RecChain_ARGS
    RecChain_ARGS = {'PID': {'folder': pdt_folder, 'RecName': PDTName}}
    # RecChain_ARGS
    RecInfo_ARGS = {
        RootRec: {'folder': rec_folder, 'RecName': RootRec,   'Columns': 'ALL'}, 
        RecNameTpt: {'folder': tpt_folder, 'RecName': RecNameTpt, 'Columns': 'ALL'},
    }
    assert RecNameTpt.split('_')[0] == RecName
    
    df_RecDB_v1 = df_RecDB_Store.get(RecName, pd.DataFrame(columns = ['PID', 'Group']))
    
    # check whether the group information is store in the df_RecBD_Store.
    # Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    if Group not in df_RecDB_v1['Group'].unique() : 
        df_RecDB_v2 = get_group_info(Group, RecChain_ARGS, RecInfo_ARGS)
        df_RecDB = pd.concat([df_RecDB_v1, df_RecDB_v2]).reset_index(drop = True)
    else:
        df_RecDB = df_RecDB_v1  
        
    df_RecDB_Store[RecName] = df_RecDB
    # attention: PID might not be in df_RecDB, 
    #            it is because PID doesn't have any records to contribute to that Group.
    return RecName, df_RecDB, df_RecDB_Store




def get_CkpdName_RecTable(PDTInfo, CkpdName, DBInfo, RecNameTpt, cols = None):
    CkptPeriod = PDTInfo[CkpdName].iloc[0]

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



def add_RelativeDT_to_dfCkpdRec(df_CkpdRec, PredDT):
    if type(df_CkpdRec) != type(pd.DataFrame()): return None
    df = df_CkpdRec # .copy()
    df['5Min'] = ((PredDT - df['DT']).dt.total_seconds()/(60 * 5)).astype(int)
    df['H'] = (df['5Min']/12).astype(int)
    df['D'] = (df['H']/24).astype(int)
    df['W'] = (df['D']/7).astype(int)
    df['M'] = (df['W']/4).astype(int)
    return df


def process_CONFIG_CkpdRec_of_PDTInfo(Case_C, CkpdName, CONFIG_RecDB, 
                                             df_RecDB_Store):
    
    PDTInfo = Case_C.copy()
    
    # -------- 2. CkpdName.RecName
    RecName =  CONFIG_RecDB['RecName']
    RecNameTpt =  CONFIG_RecDB['RecNameTpt']
    CkpdRec = CkpdName + '.' + RecName
    
    df_RecDB = df_RecDB_Store[RecName] # <---- Case_C's PID's RecName must be in df_RecDB_Store
    P_RecDB = df_RecDB[df_RecDB['PID'] == PDTInfo['PID']].iloc[0] # <----- P_RecDB could be empty.
    df = get_CkpdName_RecTable(PDTInfo, CkpdName, P_RecDB, RecNameTpt)

    df = add_RelativeDT_to_dfCkpdRec(df, Case_C['PredDT'])
    PDTInfo[CkpdRec] = df
    
    Case_CR = PDTInfo
    return Case_CR



