import pandas as pd
from recfldgrn.loadtools import get_df_bucket_from_settings
from recfldgrn.datapoint import convert_PID_to_PIDgroup


def get_group_info(Group, RecChain_ARGS, RecInfo_ARGS):
    bucket_file = Group + '.p'
    df = get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS)
    df['Group'] = Group
    # if type(PID_List) == list: df = df[df['PID'].isin(PID_List)].reset_index(drop = True)
    return df

def get_CkpdName_RecTable(PDTInfo, CkpdName, DBInfo, RecName, cols = None):
    CkptPeriod = PDTInfo[CkpdName].iloc[0]
    
    if RecName not in list(DBInfo.keys()): return None
    df = DBInfo[RecName]
    if type(df) != type(pd.DataFrame()) or len(df) == 0: return None
    
    DT_cols = [i for i in df.columns if 'DT' in i]
    
    # TODO consider a special case that you will split
    if 'DT_s' in DT_cols and 'DT_e' in DT_cols: pass 
    
    # print(df.columns)
    DT_col = DT_cols[0]
    df = df.copy().rename(columns = {DT_col: 'DT'})
    index_s = df['DT'] >= CkptPeriod['DT_start']
    index_e = df['DT'] <  CkptPeriod['DT_end'] # not <=
    df = df.loc[index_s & index_e].reset_index(drop = True)
    if len(df) == 0: return None
    df['PredDT'] = PDTInfo['PredDT']
    if type(cols) != list: cols = list(df.columns)
    cols = ['PID', 'PredDT'] + [i for i in cols if i not in ['PID', 'PredDT']]
    # print(cols)
    df = df[cols]# .copy()
    return df


def get_df_RecDB_of_PDTInfo(Case_C, CONFIG_PDT, CONFIG_RecDB, df_RecDB_Store, GROUP_RANGE_SIZE):
    
    PDTInfo = Case_C.copy()
    
    # CONFIG_PDT
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    # df_PDT['Group'] = df_PDT['PID'].apply(lambda x: convert_PID_to_PIDgroup(x, GROUP_RANGE_SIZE))
    Group = convert_PID_to_PIDgroup(PDTInfo['PID'], GROUP_RANGE_SIZE)
    
    
    RecName =  CONFIG_RecDB['RecName'] # 'CGM5Min'
    rec_folder = CONFIG_RecDB['rec_folder'] # '1-Data_RFG/1-RecFld/' # folder saving the records
    Columns = CONFIG_RecDB['Columns']

    RecChain_ARGS = {'PID': {'folder': pdt_folder, 'RecName': PDTName}}
    RecInfo_ARGS = {RecName : {'folder': rec_folder, 'RecName': RecName, 'Columns': Columns}}
    
    df_RecDB_v1 = df_RecDB_Store.get(RecName, pd.DataFrame(columns = ['PID', 'Group']))
    
    # check whether the group information is store in the df_RecBD_Store.
    # Group = convert_PID_to_PIDgroup(PDTInfo['PID'], GROUP_RANGE_SIZE)
    if Group not in df_RecDB_v1['Group'].unique() : 
        df_RecDB_v2 = get_group_info(Group, RecChain_ARGS, RecInfo_ARGS)
        df_RecDB = pd.concat([df_RecDB_v1, df_RecDB_v2]).reset_index(drop = True)
    else:
        df_RecDB = df_RecDB_v1  
        
    df_RecDB_Store[RecName] = df_RecDB
    # attention: PID might not be in df_RecDB, 
    #            it is because PID doesn't have any records to contribute to that Group.
    return RecName, df_RecDB, df_RecDB_Store


def process_CONFIG_CkpdRec_of_PDTInfo(Case_C, CkpdName, CONFIG_RecDB, df_RecDB_Store):
    
    PDTInfo = Case_C.copy()
    RecName =  CONFIG_RecDB['RecName']
    CkpdRec = CkpdName + '.' + RecName
    
    df_RecDB = df_RecDB_Store[RecName] # <---- Case_C's PID's RecName must be in df_RecDB_Store
    P_RecDB = df_RecDB[df_RecDB['PID'] == PDTInfo['PID']].iloc[0] # <----- P_RecDB could be empty.
    df = get_CkpdName_RecTable(PDTInfo, CkpdName, P_RecDB, RecName)

    # df = add_RelativeDT_to_dfCkpdRec(df, Case_C['PredDT'])
    PDTInfo[CkpdRec] = df

    Case_CR = PDTInfo
    return Case_CR
