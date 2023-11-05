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

