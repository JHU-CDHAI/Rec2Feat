import pandas as pd
from functools import reduce
import numpy as np

from .cmpfn import FlattenFunction, ConvertKey2ValueFunction, MergeLeftAndRight
from .cmpfn import convert_df_grnseq_to_flatten, convert_df_flatten_to_grnseq
from .cmpfn import add_sfx_as_new_locgrnseq_to_dfrec

from recfldgrn.graintools import get_highorder_input_idx

def get_CkpdRecFltTknCmp_Name(CkpdRecFltTkn, CompressArgs, prefix_layer_cols, focal_layer_cols):
    CkpdName, RecName, FilterName, SynFldGrn = CkpdRecFltTkn.split('.')
    prefix = '-'.join(prefix_layer_cols).replace('PDTID', 'CP')
    method = ''.join(reversed([i.split('In')[-1] +'.'+ Args['method']  for i, Args in CompressArgs.items()]))
    SynFldTknCmp = '-'.join([prefix, method]) + SynFldGrn if len(prefix) > 0 else method + SynFldGrn
    CkpdRecFltTknCmp = '.'.join([CkpdName, RecName, FilterName, SynFldTknCmp])
    return CkpdRecFltTknCmp


def compress_df_flatten_with_cmpfn(df_flatten, sourceInTarget, flatten_fn, prefix_cols):
    source, target = sourceInTarget.split('In')
    target_idx = prefix_cols.index(target)
    new_prefix_cols = prefix_cols[: target_idx + 1]
    
    drop_columns = [i for i in df_flatten.columns if i in prefix_cols[target_idx + 1:]]
    df = df_flatten.drop(columns = drop_columns)
    grn_cols = [i for i in df.columns if i not in new_prefix_cols]
    
    fn = FlattenFunction(flatten_fn, grn_cols)
    df_cmp = df.groupby(new_prefix_cols).apply(fn).reset_index()

    df_recnum = df[target].value_counts().reset_index()
    df_recnum.columns = [target, sourceInTarget]
    
    df_cmpfinal = pd.merge(df_cmp, df_recnum, on = target)
    return df_cmpfinal, new_prefix_cols


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


def convert_df_compressed_to_DPLevel_tensor(df_cmp, SynFldVocab, CkpdRecFltGrnCmp, 
                                            prefix_layer_cols, focal_layer_cols):
    df = df_cmp # .copy()
    # CkpdRecFltGrnCmpFeat = 'Case.' + CkpdRecFltGrnCmp

    if CkpdRecFltGrnCmp+'_key' in df.columns:
        df[CkpdRecFltGrnCmp+'_key'] = df[CkpdRecFltGrnCmp+'_key'].apply(ConvertKey2ValueFunction(SynFldVocab['grn2idx']))
        df = df.rename(columns = {CkpdRecFltGrnCmp+'_key': CkpdRecFltGrnCmp+'_keyidx'})
    
    if CkpdRecFltGrnCmp+'_rfg' in df.columns:
        df[CkpdRecFltGrnCmp+'_rfg'] = df[CkpdRecFltGrnCmp+'_rfg'].apply(ConvertKey2ValueFunction(SynFldVocab['rfg2idx']))
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
        df_tensor_fnl = reduce(MergeLeftAndRight(), df_list)
        
    # df_tensor_fnl.columns = [CkpdRecFltGrnCmpFeat + '_' + i.split('_')[-1] if 'Grn' in i else i for i in df_tensor_fnl.columns]
    df_tensor_fnl.columns = [CkpdRecFltGrnCmp + '_' + i.split('_')[-1] if 'Grn' in i else i for i in df_tensor_fnl.columns]
    
    df_tensor_fnl = df_tensor_fnl.reset_index(drop = True)
    return df_tensor_fnl 


def process_CONFIG_CMP_of_PDTInfoCRFT(Case_CRFT, CkpdRecFltTkn, CONFIG_CMP, UTILS_CMP, SynFldVocabNew):
    PDTInfo = Case_CRFT# .copy()
    PID, PredDT = Case_CRFT['PID'], Case_CRFT['PredDT']
    
    Ckpd, RecName, FilterName, SynFldTkn = CkpdRecFltTkn.split('.')
    
    CompressArgs = CONFIG_CMP['CompressArgs']
    prefix_layer_cols = CONFIG_CMP['Layer_Args']['prefix_layer_cols']
    focal_layer_cols = CONFIG_CMP['Layer_Args']['focal_layer_cols']
    method_to_fn = UTILS_CMP['method_to_fn'] 
    
    
    CkpdRecFltTknCmp = get_CkpdRecFltTknCmp_Name(CkpdRecFltTkn, CompressArgs, prefix_layer_cols, focal_layer_cols)
    df_CkpdRecFltGrn = Case_CRFT[CkpdRecFltTkn]# .copy()
    # TODO: df_CkpdRecFltGrn might contains no ['PID', 'PredDT', 'DT', 'R']
    df_CkpdRecFltGrn = df_CkpdRecFltGrn.drop(columns = ['PID', 'PredDT', 'DT', 'R'])
    df = df_CkpdRecFltGrn
    df_cmp = compress_dfrec_with_CompressArgs(df, CompressArgs, SynFldTkn, CkpdRecFltTknCmp, SynFldVocabNew, method_to_fn)
    df_tensor_fnl = convert_df_compressed_to_DPLevel_tensor(df_cmp, SynFldVocabNew, 
                                                            CkpdRecFltTknCmp, 
                                                            prefix_layer_cols, focal_layer_cols)
    
    df_tensor_fnl['PID'] = PID
    df_tensor_fnl['PredDT'] = PredDT
    df_tensor_fnl = df_tensor_fnl.drop(columns = ['CP'])
    PDTInfo[CkpdRecFltTknCmp] = df_tensor_fnl
    Case_CRFTC = PDTInfo
    return Case_CRFTC 

