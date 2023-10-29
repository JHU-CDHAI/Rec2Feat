import pandas as pd
from recfldgrn.datapoint import convert_PID_to_PIDgroup
from .utils_ckpdrec import get_info_from_settings, get_Ckpd_from_PredDT, get_CkpdName_RecTable
from .utils_recflt import filter_CkpdRec_fn
from .utils_grnseq import get_df_ckpdrecfltgrn, generate_dfempty
from .utils_vocab import build_SynFldVocab, get_update_SynFldVocab_and_GrnSeqName
from .utils_cmp import compress_dfrec_with_CompressArgs, convert_df_compressed_to_DPLevel_tensor
from .utils_fn import UTILS_Flt, UTILS_CMP

IDNAME = 'PID'
RANGE_SIZE = 100
UNK_TOKEN = '<UNK>'

def process_df_PDT_with_CONFIG_CkpdRecNameFlt(df_PDT, CONFIG_PDT, CONFIG_CkpdRecNameFlt, UTILS_Flt = UTILS_Flt, RANGE_SIZE = RANGE_SIZE):
    df_PDT = df_PDT.copy()
    CONFIG_Ckpd = CONFIG_CkpdRecNameFlt['CONFIG_Ckpd']
    CONFIG_RecDB = CONFIG_CkpdRecNameFlt['CONFIG_RecDB']
    CONFIG_Flt = CONFIG_CkpdRecNameFlt['CONFIG_Flt']
    
    # -------- initial df_PDT
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    df_PDT['Group'] = df_PDT['PID'].apply(lambda x: convert_PID_to_PIDgroup(x, RANGE_SIZE))
    # result_dict['df_PDT'] = df_PDT
    
    # -------- initial df_RecDB
    rec_folder = CONFIG_RecDB['rec_folder'] 
    RootRec =  CONFIG_RecDB['RootRec']
    tpt_folder =  CONFIG_RecDB['tpt_folder']
    RecName =  CONFIG_RecDB['RecName']
    RecNameTpt =  CONFIG_RecDB['RecNameTpt']
    RecChain_ARGS = {
        'PID': {'folder': pdt_folder, 'RecName': PDTName}, 
    }
    RecInfo_ARGS = {
        RootRec: {'folder': rec_folder, 'RecName': RootRec,   'Columns': 'ALL'}, 
        RecNameTpt: {'folder': tpt_folder, 'RecName': RecNameTpt, 'Columns': 'ALL'},
    }
    assert RecNameTpt.split('_')[0] == RecName
    df_group2PID = df_PDT[['Group', 'PID']].groupby('Group').apply(lambda x: list(x['PID'].unique())).reset_index()
    df_group2PID = df_group2PID.rename(columns = {0: 'PID_List'})
    s = df_group2PID.apply(lambda rec: get_info_from_settings(rec['Group'], rec['PID_List'], RecChain_ARGS, RecInfo_ARGS), axis = 1)
    df_RecDB = pd.concat(s.to_list()).reset_index(drop = True)
    # result_dict['df_RecDB'] = df_RecDB
    
    # -------- initial CkpdName
    CkpdName = CONFIG_Ckpd['CkpdName'] # 'Bf1D'
    Ckpd_ARGS = {
        'CkpdName':  CONFIG_Ckpd['CkpdName'], 
        'DistStartToPredDT': CONFIG_Ckpd['DistStartToPredDT'],
        'DistEndToPredDT':  CONFIG_Ckpd['DistEndToPredDT'],
        'TimeUnit':  CONFIG_Ckpd['TimeUnit'], 
    }
    CkpdRec = CkpdName + '.' + RecName
    df_PDT[CkpdName] = df_PDT['PredDT'].apply(lambda PredDT: get_Ckpd_from_PredDT(PredDT, **Ckpd_ARGS))
    df_PDT[CkpdRec] = df_PDT.apply(lambda PDTInfo: get_CkpdName_RecTable(PDTInfo, CkpdName, 
                                                     df_RecDB[df_RecDB['PID'] == PDTInfo['PID']].iloc[0],
                                                     RecNameTpt), axis = 1)
    
    # result_dict['CkpdRec'] = CkpdRec
    
    # -------- initial Filter
    FilterName = CONFIG_Flt['FilterName'] # 'Whl'
    CkpdRecFlt = CkpdRec + '.' + FilterName
    FilterName2FilterFunc = UTILS_Flt['FilterName2FilterFunc']
    FilterName2Share = UTILS_Flt['FilterName2Share']
    RecName2ThresConfig = UTILS_Flt['RecName2ThresConfig']
    CkpdRecFilter_ARGS = {
        'FilterName': FilterName,
        'FilterFunc': FilterName2FilterFunc[FilterName],
        'ThresShare': FilterName2Share[FilterName],
        'ThresConfig': RecName2ThresConfig[RecNameTpt.split('_')[0]]
    }
    df_PDT[CkpdRecFlt] = df_PDT.apply(lambda x: filter_CkpdRec_fn(x['PredDT'], x[CkpdRec], **CkpdRecFilter_ARGS), axis = 1)

    # result_dict['CkpdRecFlt'] = CkpdRecFlt
    # result_dict['df_PDT'] = df_PDT
    df_PDT_CkpdRecFlt = df_PDT
    return df_PDT_CkpdRecFlt, CkpdRecFlt


def process_df_PDT_CkpdRecFlt_with_CONFIG_GrnSeq(df_PDT_CkpdRecFlt, CONFIG_PDT, CONFIG_GrnSeq):
    df_PDT = df_PDT_CkpdRecFlt.copy()
    
    CkpdRecFlt = df_PDT.columns[-1]
    Ckpd, RecName, FilterName = CkpdRecFlt.split('.')
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    
    # -------- CONFIG_GrnSeq
    fldgrn_folder = CONFIG_GrnSeq['fldgrn_folder']
    fldgrnv_folder = CONFIG_GrnSeq['fldgrnv_folder']
    GrnName = CONFIG_GrnSeq['GrnName']
    RecFldGrn_List = CONFIG_GrnSeq['RecFldGrn_List'] 
    RDTCal_rfg = CONFIG_GrnSeq['RDTCal_rfg'] 
    SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
    CkpdRecFltGrn = CkpdRecFlt + '.' + SynFldGrn
    GrnSeqChain_ARGS = {'PID': {'folder': pdt_folder, 'RecName': PDTName}}
    
    GrnSeqInfo_ARGS = {}
    for fld in RecFldGrn_List:
        GrnSeqInfo_ARGS[fld] = {'folder': fldgrn_folder,  'RecName': fld,  'Columns': 'ALL'}
    
    df_group2PID = df_PDT[['Group', 'PID']].groupby('Group').apply(lambda x: list(x['PID'].unique())).reset_index()
    df_group2PID = df_group2PID.rename(columns = {0: 'PID_List'})
    s = df_group2PID.apply(lambda rec: get_info_from_settings(rec['Group'], rec['PID_List'], GrnSeqChain_ARGS, GrnSeqInfo_ARGS), axis = 1)
    df_GrnDB = pd.concat(s.to_list()).reset_index(drop = True)
    df_PDT[CkpdRecFltGrn] = df_PDT.apply(lambda PDTInfo: get_df_ckpdrecfltgrn(PDTInfo[CkpdRecFlt], 
                                                          df_GrnDB[df_GrnDB['PID'] == PDTInfo['PID']].iloc[0], 
                                                          RecFldGrn_List, RDTCal_rfg, 
                                                          RecName, SynFldGrn), axis = 1)


    df_PDT[CkpdRecFltGrn] = df_PDT.apply(lambda x: generate_dfempty(x['PID'], x['PredDT'], CkpdRecFltGrn, UNK_TOKEN)
                                         if type(x[CkpdRecFltGrn]) != type(pd.DataFrame()) else x[CkpdRecFltGrn], axis = 1)
    
    SynFldVocab = build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, UNK_TOKEN)  
    
    df_PDT_CkpdRecFltGrn = df_PDT
    CkpdRecFltTkn = CkpdRecFltGrn
    return df_PDT_CkpdRecFltGrn, CkpdRecFltTkn, SynFldVocab


def process_df_PDT_CkpdRecFltGrn_with_CONFIG_CMP(df_PDT_CkpdRecFltTkn, CONFIG_PDT, CONFIG_GrnSeq, CONFIG_CMP, UTILS_CMP = UTILS_CMP, UNK_TOKEN = UNK_TOKEN):
    df_PDT = df_PDT_CkpdRecFltTkn.copy()
    
    CkpdRecFlt = df_PDT.columns[-2]
    CkpdRecFltGrn =  df_PDT.columns[-1]
    CkpdName, RecName, FilterName = CkpdRecFlt.split('.')

    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    
    # -------- CONFIG_CMP
    fldgrn_folder = CONFIG_GrnSeq['fldgrn_folder']
    fldgrnv_folder = CONFIG_GrnSeq['fldgrnv_folder']
    GrnName = CONFIG_GrnSeq['GrnName']
    RecFldGrn_List = CONFIG_GrnSeq['RecFldGrn_List']  
    RDTCal_rfg = CONFIG_GrnSeq['RDTCal_rfg'] 
    SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
    
    
    method_to_fn = UTILS_CMP['method_to_fn']
    CompressArgs = CONFIG_CMP['CompressArgs']
    prefix_layer_cols = CONFIG_CMP['prefix_layer_cols']
    focal_layer_cols = CONFIG_CMP['focal_layer_cols']
    
    SynFldVocab = build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, UNK_TOKEN)  
    
    SynFldVocabNew, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat = \
            get_update_SynFldVocab_and_GrnSeqName(PDTName, CkpdName, RecName, FilterName, SynFldGrn, 
                                                  SynFldVocab, CompressArgs, 
                                                  prefix_layer_cols, focal_layer_cols)


    df_PDT[CkpdRecFltGrnCmp] = df_PDT[CkpdRecFltGrn].apply(lambda df: compress_dfrec_with_CompressArgs(df, CompressArgs, SynFldGrn, CkpdRecFltGrnCmp, SynFldVocabNew, method_to_fn))
    df_PDT[CkpdRecFltGrnCmpFeat] = df_PDT[CkpdRecFltGrnCmp].apply(lambda df_cmp: 
                                        convert_df_compressed_to_DPLevel_tensor(
                                                                      df_cmp, SynFldVocabNew, 
                                                                      CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat, 
                                                                      prefix_layer_cols, focal_layer_cols))
    
    df_PDT_CkpdRecFltTknCmp = df_PDT
    SynFldVocab = SynFldVocabNew
    return df_PDT_CkpdRecFltTknCmp, SynFldVocab, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat
