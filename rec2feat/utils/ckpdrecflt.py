import pandas as pd

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

def filter_CkpdRec_fn(PredDT, df_CkpdRec, FilterName, FilterFunc, ThresShare, ThresConfig):
    df = df_CkpdRec
    
    if type(df_CkpdRec) != type(pd.DataFrame()): return None
    # 1. add relative DT
    # df = add_RelativeDT_to_dfCkpdRec(df, PredDT)

    if FilterName == 'Whl': return df

    # 2. filter df with filterfn
    df = filter_with_FilterFunc(df, FilterFunc)
    # 3. threshold sharing
    df = filter_with_Threshold_Args(df, ThresConfig, ThresShare)

    if len(df) == 0: return None
    return df

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
    
    if FilterName == 'Whl': 
        PDTInfo[CkpdRecFlt] = PDTInfo[CkpdRec]
    else:
        PDTInfo[CkpdRecFlt] = filter_CkpdRec_fn(PredDT, PDTInfo[CkpdRec], **CkpdRecFilter_ARGS)
    
    Case_CRF = PDTInfo
    return Case_CRF