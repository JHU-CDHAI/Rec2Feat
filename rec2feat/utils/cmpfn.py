import numpy as np
import pandas as pd

class ConvertKey2ValueFunction:
    def __init__(self, key2value):
        self.key2value = key2value
    
    def __call__(self, x):
        return [self.key2value[i] for i in x]
    
class GenerateFldTknLocIdx:
    def __init__(self, col, SynFldGrn):
        self.col = col
        self.SynFldGrn = SynFldGrn
    
    def __call__(self, x):
        return [x[self.col]] * len(x[self.SynFldGrn+'_'+'key'])
    

class DictZipFunction:
    def __init__(self, SynFldGrn):
        self.SynFldGrn = SynFldGrn
    def __call__(self, x):
        SynFldGrn = self.SynFldGrn
        d = dict(zip(x[SynFldGrn +'_key'], x[SynFldGrn +'_wgt']))      
        return d
    
class FlattenFunction:
    def __init__(self, flatten_fn, grn_cols):
        self.flatten_fn = flatten_fn
        self.grn_cols = grn_cols
    
    def __call__(self, df_group):
        # Here, implement the logic that was originally in the flatten_fn
        # For the sake of the example, let's say flatten_fn does some kind of flattening
        # and returns a pandas Series or DataFrame.
        flattened_data = self.flatten_fn(df_group[self.grn_cols])
        return flattened_data

class ConvertKey2ValueFunction:
    def __init__(self, key2value):
        self.key2value = key2value
    
    def __call__(self, x):
        return [self.key2value[i] for i in x]
    

class MergeLeftAndRight:
    def __init__(self):
        pass
    def __call__(self, left, right):
        return pd.merge(left, right)
    

def convert_rfg_to_keyInrfg(data):
    unique_labels, inverse_indices = np.unique(data, return_inverse=True)
    result = np.zeros(len(data), dtype=int)
    for i, label in enumerate(unique_labels):
        mask = (inverse_indices == i)
        result[mask] = np.arange(np.sum(mask))
    return list(result)


def add_sfx_as_new_locgrnseq_to_dfrec(df, SynFldGrn, SynFldVocab, sfx_list):
    for sfx in sfx_list:
        
        if sfx == 'rfg':
            df[SynFldGrn+'_'+'rfg'] = df[SynFldGrn+'_'+'key'].apply(ConvertKey2ValueFunction(SynFldVocab['grn2rfg']))
        
        elif sfx == 'keyInrfg':
            df[SynFldGrn+'_'+'keyInrfg'] = df[SynFldGrn+'_'+'rfg'].apply(convert_rfg_to_keyInrfg)
        
        elif sfx.replace('InCP', '') in df.columns:
            df[SynFldGrn+'_'+sfx] = df.apply(GenerateFldTknLocIdx(sfx.replace('InCP', ''), SynFldGrn), axis = 1)
        
        elif 'In' in sfx:
            sourceInTarget = sfx
            source, target = sourceInTarget.split('In')
            df_st = df[[source, target]].drop_duplicates()
            df_st[sourceInTarget] = df_st.groupby(target).cumcount()
            df = pd.merge(df, df_st)
            df[SynFldGrn+'_'+sourceInTarget] = df.apply(GenerateFldTknLocIdx(sourceInTarget, SynFldGrn), axis = 1)
            df = df.drop(columns = [sourceInTarget])
        
    return df

def convert_df_grnseq_to_flatten(df, SynFldGrn, SynFldVocab):
    s = df.apply(DictZipFunction(SynFldGrn), axis = 1)
    dfx = pd.DataFrame(s.to_list())
    
    idx2grn = [i for i in SynFldVocab['idx2grn'] if i in dfx.columns]
    prefix_cols = [i for i in df.columns if SynFldGrn not in i]
    dfx = dfx[idx2grn].fillna(0)
    dfx = pd.concat([df[prefix_cols], dfx], axis = 1)
    return dfx, prefix_cols

def x_positive_to_dict(x): return x[x>0].to_dict()
def x_keys_to_list(x): return list(x.keys())
def x_values_to_list(x): return list(x.values())

def convert_df_flatten_to_grnseq(df_flatten, SynFldGrn, SynFldVocab):
    idx2grn = [i for i in SynFldVocab['idx2grn'] if i in df_flatten.columns]
    dfx = df_flatten[[i for i in df_flatten.columns if i not in idx2grn]].reset_index(drop = True)
    dfx[SynFldGrn] = df_flatten[idx2grn].apply(x_positive_to_dict, axis = 1)
    dfx[f'{SynFldGrn}_key'] = dfx[SynFldGrn].apply(x_keys_to_list)
    dfx[f'{SynFldGrn}_wgt'] = dfx[SynFldGrn].apply(x_values_to_list)
    dfx = dfx.drop(columns = [SynFldGrn])
    return dfx