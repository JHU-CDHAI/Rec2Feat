import pandas as pd

def add_RelativeDT_to_dfCkpdRec(df_CkpdRec, PredDT):
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


def filter_CkpdRec_fn(PredDT, df_CkpdRec, FilterName, FilterFunc, ThresShare, ThresConfig):
    df = df_CkpdRec
    if type(df_CkpdRec) != type(pd.DataFrame()): return None

    # 1. add relative DT
    df = add_RelativeDT_to_dfCkpdRec(df, PredDT)
    # 2. filter df with filterfn
    df = filter_with_FilterFunc(df, FilterFunc)
    # 3. threshold sharing
    df = filter_with_Threshold_Args(df, ThresConfig, ThresShare)
    
    if len(df) == 0: return None
    return df



######################## Hyper Paramters
FilterName2FilterFunc = {
    'Whl': lambda x: True, 
    'Morning': lambda x: x['DT'].hour in [6,7,8,9,10,11],
    'Afternoon': lambda x: x['DT'].hour in [12, 13, 14, 15, 16, 17],
    'Evening': lambda x: x['DT'].hour in [18, 19, 20, 21, 22, 23],
    'Night': lambda x: x['DT'].hour in [0, 1, 2, 3, 4, 5],
}

FilterName2Share = {
    'Whl': 1, 
    'Morning': 0.25,
    'Afternoon': 0.25,
    'Evening': 0.25,
    'Night': 0.25,
}

RecName2ThresConfig = {
    'CGM5Min': {
        'RInDT': {'Threshold': 0,       'MaxNum': None,}, 
        'DTInD': {'Threshold': 288*0.7, 'MaxNum': 288, }, 
        'DInCP': {'Threshold': 0,       'MaxNum': None, },
    }
}
########################
