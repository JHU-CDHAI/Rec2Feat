import pandas as pd


def add_RelativeDT_to_dfCkpdRec(df_CkpdRec, PredDT):
    if type(df_CkpdRec) != type(pd.DataFrame()): return None
    df = df_CkpdRec # .copy()
    df['5Min'] = ((PredDT - df['DT']).dt.total_seconds()/(60 * 5)).astype(int)
    df['H'] = (df['5Min']/12).astype(int)
    df['D'] = (df['H']/24).astype(int)
    df['W'] = (df['D']/7).astype(int)
    df['M'] = (df['W']/4).astype(int)
    return df

def filter_with_FilterFunc(df, FilterFunc):
    df = df[df.apply(lambda rec: FilterFunc(rec), axis = 1)]
    return df

def filter_with_Threshold_Args(df, ThresConfig, ThresShare):
    # df = df_bfrec_selected
    for sourceInTarget, ARGS in ThresConfig.items():
        # sourceInTarget = 'DTInD'
        # ARGS = THRESHOLD_ARGS[sourceInTarget]
        Threshold = ARGS['Threshold'] * ThresShare
        if Threshold == 0: continue
        source, target = sourceInTarget.split('In')
        dfx = df[[target, source]].drop_duplicates()
        dfx = dfx[target].value_counts().reset_index()
        dfx.columns = [target, sourceInTarget]
        dfx = dfx[dfx[sourceInTarget] >= Threshold]# [target].to_list()
        target_valid_list = dfx[target].to_list()
        df = df[df[target].isin(target_valid_list)].reset_index(drop = True)
    return df

def filter_CkpdRec_fn(PredDT, RecName, df_CkpdRec, 
                      FilterName, FilterFunc, 
                      ThresShare, ThresConfig):
    
    cols = ['PID', 'PredDT', RecName + 'ID', 'DT']  
    
    df = df_CkpdRec
    if type(df) != type(pd.DataFrame()): return None

    # 1. filter with DT or Value
    if FilterName != 'Whl': 
        df = filter_with_FilterFunc(df, FilterFunc).reset_index(drop = True)

    # filter with Threshold Config
    if len(ThresConfig) == 0: return df[cols].reset_index(drop = True)

    # 2. add relative DT
    df = add_RelativeDT_to_dfCkpdRec(df, PredDT)
    
    # 3. threshold sharing
    df = filter_with_Threshold_Args(df, ThresConfig, ThresShare)

    if len(df) == 0: return None
    return df[cols].reset_index(drop = True)


def process_CONFIG_CkpdRecFlt_of_PDTInfo(Case_CR, CkpdRec, CONFIG_Flt, UTILS_Flt):
    
    PDTInfo = Case_CR.copy()
    CkpdName, RecName = CkpdRec.split('.')
    
    FilterName = CONFIG_Flt['FilterName'] # 'Whl'
    FilterName2FilterFunc = UTILS_Flt['FilterName2FilterFunc']
    FilterName2Share = UTILS_Flt['FilterName2Share']
    RecName2ThresConfig = UTILS_Flt['RecName2ThresConfig']
    CkpdRecFilter_ARGS = {
        'FilterName': FilterName,
        'FilterFunc': FilterName2FilterFunc[FilterName],
        'ThresShare': FilterName2Share[FilterName],
        'ThresConfig': RecName2ThresConfig[RecName]
    }
    CkpdRecFlt = CkpdRec + '.' + FilterName
    PredDT = PDTInfo['PredDT']
    PDTInfo[CkpdRecFlt] = filter_CkpdRec_fn(PredDT, RecName, PDTInfo[CkpdRec], **CkpdRecFilter_ARGS)
    Case_CRF = PDTInfo
    return Case_CRF
