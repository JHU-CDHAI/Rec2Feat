import os
import pandas as pd
from functools import reduce
from recfldgrn.datapoint import convert_PID_to_PIDgroup

def get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS):
    L = []
    RecNameID_Chain = [i for i in RecChain_ARGS]
    for idx, RID in enumerate(RecChain_ARGS):
        RecInfo = RecChain_ARGS[RID]
        folder, RecName = RecInfo['folder'], RecInfo['RecName']
        df = pd.read_pickle(os.path.join(folder, RecName, bucket_file))
        df = df[RecNameID_Chain[:idx+ 1]].astype(str).drop_duplicates()
        L.append(df)
    df_prefix = reduce(lambda left, right: pd.merge(left, right, how = 'left'), L)
    
    # fill the missing Rec with missing ID.
    for RID in RecNameID_Chain:
        s = 'M' + pd.Series(df_prefix.index).astype(str)
        df_prefix[RID] = df_prefix[RID].fillna(s)
    assert len(df_prefix) != 0

    # 2 ----------------- get df_data
    RecLevelID = RecNameID_Chain[-1] # in most case,  be PID.

    df_whole = df_prefix
    # you can use a single version: only one record in RecInfo_ARGS
    for idx, RecElmt in enumerate(RecInfo_ARGS):
        RecElmt_ARGS = RecInfo_ARGS[RecElmt]
        folder = RecElmt_ARGS['folder']
        RecName = RecElmt_ARGS['RecName']
        FldList = RecElmt_ARGS['Columns']
        
        # check whether a file exists.
        path = os.path.join(folder, RecName, bucket_file)
        if not os.path.exists(path): 
            print(f'empty path: {path}')
            df = pd.DataFrame(columns = [RecLevelID, RecElmt])
            df_whole = pd.merge(df_whole, df, how = 'left')
            continue 
        
        # read df
        df = pd.read_pickle(path)
        
        # select columns
        if FldList == 'ALL': FldList = list(df.columns)
        full_cols = [i for i in RecNameID_Chain if i not in FldList] + FldList
        full_cols = [i for i in full_cols if i in df.columns]
        df = df[full_cols].reset_index(drop = True)
        for RecID in RecNameID_Chain: 
            if RecID in df.columns: df[RecID] = df[RecID].astype(str)

        # downstream df_data with df_prefix if RecLevelID is not in ID
        # eg: RecLevelID from df_prefix is Encounter-level, but df is Patient-level.
        #     we want to convert df has a RecLevelID. 
        if RecLevelID not in df.columns:
            on_cols = [i for i in df_prefix.columns if i in df.columns]
            df = pd.merge(df_prefix, df, on = on_cols, how = 'left')

        # upstream: now df has the RecLevelID, we want to group it by RecLevelID and merge to df_prefix.
        # df: ['RecLevelID' and 'RecElmt']
        df = pd.DataFrame([{RecLevelID: RecLevelIDValue, RecElmt: df_input} 
                           for RecLevelIDValue, df_input in df.groupby(RecLevelID)])
        
        df_whole = pd.merge(df_whole, df, how = 'left')

    return df_whole


def get_group_info(Group, RecChain_ARGS, RecInfo_ARGS):
    bucket_file = Group + '.p'
    try:
        df = get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS)
        df['Group'] = Group
    except:
        print(bucket_file, RecChain_ARGS, RecInfo_ARGS, '<---- error in get_df_bucket_from_settings')
        df = pd.DataFrame()
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
