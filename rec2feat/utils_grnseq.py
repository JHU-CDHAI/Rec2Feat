import pandas as pd

def mapping_hour_to_MAEN(x):
    if x in [6,7,8,9,10,11]:
        return 'Morning'
    elif x in [12, 13, 14, 15, 16, 17]:
        return 'Afternoon'
    elif x in [18, 19, 20, 21, 22, 23]:
        return 'Evening'
    elif x in [0, 1, 2, 3, 4, 5]:
        return 'Night'
    else:
        raise ValueError(f'{x} is not available')

def get_DTCal_grn_fn(dt):
    # dt = rec['DT_s']
    MEAN = mapping_hour_to_MAEN(dt.hour)
    d = {
        f"Year-{dt.year}": 1, 
        f"Month-{dt.month}": 1,
        f"Date-{dt.day}": 1,
        f"MEAN-{MEAN}": 1,
        f"Hour-{dt.hour}": 1,
        f"Min-{dt.minute}": 1,
    }
    key = list([k for k in d])
    wgt = list([v for k, v in d.items()])
    output = {'key': key, 'wgt': wgt}
    return output

def add_RDTCalGrn(df, recfldgrn):
    df[recfldgrn] = df.apply(lambda rec: get_DTCal_grn_fn(rec['DT']), axis = 1)
    for i in ['key', 'wgt']: df[f'{recfldgrn}_{i}'] = df[recfldgrn].apply(lambda x: x[i])
    df = df.drop(columns = [recfldgrn])
    return df


def cat_SynFldGrn_from_RecFldGrn_List(df, SynFldGrn, RDTCal_rfg, RecFldGrn_List):
    if type(RDTCal_rfg) == str:
        RecFldGrn_List = RecFldGrn_List + [RDTCal_rfg] 
    
    suffix_list = list(set([i.split('_')[-1] for i in df.columns if 'Grn' in i]))
    suffix_list = [i for i in ['key', 'wgt', 'rfg', 'keyInrfg'] if i in suffix_list]
    for suffix in suffix_list:
        RecFldGrn_sfx_cols = [rfg+'_'+suffix for rfg in RecFldGrn_List] 
        df[SynFldGrn+'_'+suffix] = df[RecFldGrn_sfx_cols].sum(axis = 1) 
    SynFldGrn_cols = [i for i in df.columns if SynFldGrn in i]
    
    df['CP'] = 0
    DTRel_cols = ['CP', 'M', 'W', 'D', 'H', '5Min']
    DTRel_cols = [i for i in DTRel_cols if i in df.columns]
    df = df[['PID', 'PredDT'] + DTRel_cols + ['DT', 'R'] + SynFldGrn_cols].reset_index(drop = True)
    return df


def get_df_ckpdrecfltgrn(df_ckpdrecflt, GrnDBInfo, RecFldGrn_List, RDTCal_rfg, RecName, SynFldGrn):
    if type(df_ckpdrecflt) != type(pd.DataFrame()): return None
    
    df = df_ckpdrecflt# .copy()
    if type(RDTCal_rfg) == str: 
        df = add_RDTCalGrn(df, RDTCal_rfg)

    for rfg in RecFldGrn_List:
        df_RFG = GrnDBInfo[rfg]
        if type(df_RFG) != type(pd.DataFrame()): continue
        df = pd.merge(df, df_RFG, how = 'left')

    df = df.rename(columns = {RecName + 'ID': 'R'})
    df['R'] = RecName + df['R'].astype(str)
    df_ckpdrecfltgrn = cat_SynFldGrn_from_RecFldGrn_List(df, SynFldGrn, RDTCal_rfg, RecFldGrn_List)
    
    return df_ckpdrecfltgrn

def generate_dfempty(PID, PredDT, CkpdRecFltGrn, UNK_TOKEN):
    RecGrnName = CkpdRecFltGrn.split('.')[-1]
    DTCal_cols = ['CP', 'M', 'W', 'D', 'H', '5Min']
    keys = ['PID', 'PredDT'] + DTCal_cols + ['DT', 'R'] + [RecGrnName+'_key', RecGrnName+'_wgt']
    values = [PID, PredDT] + [0] * len(DTCal_cols) + [None, None] + [[UNK_TOKEN], [1]]
    d = dict(zip(keys, values))
    df = pd.DataFrame([d])
    return df



def convert_df_grnseq_to_flatten(df, SynFldGrn, SynFldVocab):
    s = df.apply(lambda x: dict(zip(x[SynFldGrn +'_key'], x[SynFldGrn +'_wgt'])), axis = 1)
    dfx = pd.DataFrame(s.to_list())
    
    idx2grn = [i for i in SynFldVocab['idx2grn'] if i in dfx.columns]
    prefix_cols = [i for i in df.columns if SynFldGrn not in i]
    dfx = dfx[idx2grn].fillna(0)
    dfx = pd.concat([df[prefix_cols], dfx], axis = 1)
    return dfx, prefix_cols

def convert_df_flatten_to_grnseq(df_flatten, SynFldGrn, SynFldVocab):
    idx2grn = [i for i in SynFldVocab['idx2grn'] if i in df_flatten.columns]
    dfx = df_flatten[[i for i in df_flatten.columns if i not in idx2grn]].reset_index(drop = True)
    dfx[SynFldGrn] = df_flatten[idx2grn].apply(lambda x: x[x>0].to_dict(), axis = 1)
    dfx[f'{SynFldGrn}_key'] = dfx[SynFldGrn].apply(lambda x: list(x.keys()))
    dfx[f'{SynFldGrn}_wgt'] = dfx[SynFldGrn].apply(lambda x: list(x.values()))
    dfx = dfx.drop(columns = [SynFldGrn])
    return dfx