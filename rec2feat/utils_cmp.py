import pandas as pd
import numpy as np
from functools import reduce
from recfldgrn.graintools import get_highorder_input_idx
from .utils_grnseq import convert_df_grnseq_to_flatten, convert_df_flatten_to_grnseq

def compress_df_flatten_with_cmpfn(df_flatten, sourceInTarget, flatten_fn, prefix_cols):
    source, target = sourceInTarget.split('In')
    target_idx = prefix_cols.index(target)
    new_prefix_cols = prefix_cols[: target_idx + 1]
    
    drop_columns = [i for i in df_flatten.columns if i in prefix_cols[target_idx + 1:]]
    df = df_flatten.drop(columns = drop_columns)
    grn_cols = [i for i in df.columns if i not in new_prefix_cols]
    df_cmp = df.groupby(new_prefix_cols).apply(lambda df_group: flatten_fn(df_group[grn_cols])).reset_index()
    
    df_recnum = df[target].value_counts().reset_index()
    df_recnum.columns = [target, sourceInTarget]
    
    df_cmpfinal = pd.merge(df_cmp, df_recnum, on = target)
    return df_cmpfinal, new_prefix_cols

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
            df[SynFldGrn+'_'+'rfg'] = df[SynFldGrn+'_'+'key'].apply(lambda x: [SynFldVocab['grn2rfg'][i] for i in x])
        elif sfx == 'keyInrfg':
            df[SynFldGrn+'_'+'keyInrfg'] = df[SynFldGrn+'_'+'rfg'].apply(lambda x: convert_rfg_to_keyInrfg(x))
        elif sfx.replace('InCP', '') in df.columns:
            df[SynFldGrn+'_'+sfx] = df.apply(lambda x: [x[sfx.replace('InCP', '')]] * len(x[SynFldGrn+'_'+'key']), axis = 1)
        elif 'In' in sfx:
            sourceInTarget = sfx
            source, target = sourceInTarget.split('In')
            df_st = df[[source, target]].drop_duplicates()
            df_st[sourceInTarget] = df_st.groupby(target).cumcount()
            df = pd.merge(df, df_st)
            df[SynFldGrn+'_'+sourceInTarget] = df.apply(lambda x: [x[sourceInTarget]] * len(x[SynFldGrn+'_'+'key']), axis = 1)
        
    return df



def compress_dfrec_with_CompressArgs(df, CompressArgs, SynFldGrn, CkpdRecFltGrnCmp, SynFldVocab, method_to_fn):
    # earliest version of prefix_cols, this will be updated along the iteration.
    prefix_cols = [i for i in df.columns if 'Grn' not in i]
    
    for sourceInTarget, Args in CompressArgs.items():
        # print(sourceInTarget)
        source, target = sourceInTarget.split('In')
        
        for kw, args in Args.items(): 
            # 1. check flatten or not
            # if Args.get('flatten', False) == True:
            if kw == 'flatten' and args == True: 
                df, prefix_cols = convert_df_grnseq_to_flatten(df, SynFldGrn, SynFldVocab)
                # print(df.head())
            
            # 2. apply compress method
            # if Args['method'] == 'mean':
            if kw == 'method':
                if args not in method_to_fn: raise ValueError(f'{args} is not available')
                flatten_fn = method_to_fn[args]
                df, prefix_cols = compress_df_flatten_with_cmpfn(df, sourceInTarget, flatten_fn, prefix_cols)

            # 4. convert from flatten to grnseq or not. 
            # if Args.get('convert', False) == True:
            if kw == 'convert' and args == True: 
                df = convert_df_flatten_to_grnseq(df, SynFldGrn, SynFldVocab)

            # 5. add sfx or not
            # if type(Args.get('add_sfx', False)) == list:
            if kw == 'add_sfx' and type(args) == list: 
                df = add_sfx_as_new_locgrnseq_to_dfrec(df, SynFldGrn, SynFldVocab, args)
            
    d = {i:i.replace(SynFldGrn, CkpdRecFltGrnCmp) for i in df.columns}
    df_compressed = df.rename(columns = d)
    return df_compressed

def convert_df_compressed_to_DPLevel_tensor(df_cmp, SynFldVocabNew, 
                                            CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat, 
                                            prefix_layer_cols, focal_layer_cols):
    df = df_cmp # .copy()
    SynFldVocab = SynFldVocabNew
    
    df['CP'] = df['PID'].astype(str) + ':' + df['PredDT'].astype(str)
    df = df.rename(columns = {'CP':'PDTID'})
    
    # convert key / rfg to keyidx / rfgidx
    if CkpdRecFltGrnCmp+'_key' in df.columns:
        df[CkpdRecFltGrnCmp+'_key'] = df[CkpdRecFltGrnCmp+'_key'].apply(lambda x: [SynFldVocab['grn2idx'][i] for i in x])
        df = df.rename(columns = {CkpdRecFltGrnCmp+'_key': CkpdRecFltGrnCmp+'_keyidx'})
    if CkpdRecFltGrnCmp+'_rfg' in df.columns:
        df[CkpdRecFltGrnCmp+'_rfg'] = df[CkpdRecFltGrnCmp+'_rfg'].apply(lambda x: [SynFldVocab['rfg2idx'][i] for i in x])
        df = df.rename(columns = {CkpdRecFltGrnCmp+'_rfg': CkpdRecFltGrnCmp+'_rfgidx'})

    df_tensor = df
    tensor_cols = [i for i in df_tensor.columns if 'Grn' in i]
    if len(prefix_layer_cols) == 0:
        final_cols = focal_layer_cols + tensor_cols
        # final_cols = ['PID'] + final_cols if 'PID' not in final_cols else final_cols
        df_tensor_fnl = df_tensor[final_cols]
    else:
        df_list = []
        for recfldgrn_sfx in tensor_cols:
            df_dp = get_highorder_input_idx(df_tensor, recfldgrn_sfx, prefix_layer_cols, focal_layer_cols)
            df_list.append(df_dp)
        df_tensor_fnl = reduce(lambda left, right: pd.merge(left, right), df_list)
        
    df_tensor_fnl.columns = [CkpdRecFltGrnCmpFeat + '_' + i.split('_')[-1] if 'Grn' in i else i for i in df_tensor_fnl.columns]
    df_tensor_fnl = df_tensor_fnl.reset_index(drop = True)
    df_tensor_fnl['PID'] = df_tensor_fnl['PDTID'].apply(lambda x: int(x.split(':')[0]))

    # print(SynRecFldGrn_FullRecName)
    return df_tensor_fnl 