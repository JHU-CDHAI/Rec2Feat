import pandas as pd
from recfldgrn.datapoint import convert_PID_to_PIDgroup
from .utils_ckpdrec import get_group_info, get_Ckpd_from_PredDT, get_CkpdName_RecTable
from .utils_recflt import filter_CkpdRec_fn, add_RelativeDT_to_dfCkpdRec
from .utils_grnseq import get_df_ckpdrecfltgrn, generate_dfempty
from .utils_vocab import build_SynFldVocab, get_update_SynFldVocab_and_GrnSeqName
from .utils_cmp import compress_dfrec_with_CompressArgs, convert_df_compressed_to_DPLevel_tensor
from .utils_tool import UTILS_Flt, UTILS_CMP

IDNAME = 'PID'
RANGE_SIZE = 100
UNK_TOKEN = '<UNK>'

def get_df_RecDB_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, df_RecDB_Store, RANGE_SIZE):

    CONFIG_RecDB = CONFIG_CkpdRecNameFlt['CONFIG_RecDB']
    
    # CONFIG_PDT
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    # df_PDT['Group'] = df_PDT['PID'].apply(lambda x: convert_PID_to_PIDgroup(x, RANGE_SIZE))
    Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    
    # CONFIG_RecDB
    rec_folder = CONFIG_RecDB['rec_folder'] 
    RootRec =  CONFIG_RecDB['RootRec']
    tpt_folder =  CONFIG_RecDB['tpt_folder']
    RecName =  CONFIG_RecDB['RecName']
    RecNameTpt =  CONFIG_RecDB['RecNameTpt']
    
    # RecChain_ARGS
    RecChain_ARGS = {
        'PID': {'folder': pdt_folder, 'RecName': PDTName}, 
    }
    # RecChain_ARGS
    RecInfo_ARGS = {
        RootRec: {'folder': rec_folder, 'RecName': RootRec,   'Columns': 'ALL'}, 
        RecNameTpt: {'folder': tpt_folder, 'RecName': RecNameTpt, 'Columns': 'ALL'},
    }
    assert RecNameTpt.split('_')[0] == RecName
    
    df_RecDB_v1 = df_RecDB_Store.get(RecName, pd.DataFrame(columns = ['PID', 'Group']))
    
    # check whether the group information is store in the df_RecBD_Store.
    # Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    if Group not in df_RecDB_v1['Group'].unique() : 
        df_RecDB_v2 = get_group_info(Group, RecChain_ARGS, RecInfo_ARGS)
        df_RecDB = pd.concat([df_RecDB_v1, df_RecDB_v2]).reset_index(drop = True)
    else:
        df_RecDB = df_RecDB_v1  
        
    df_RecDB_Store[RecName] = df_RecDB
    # attention: PID might not be in df_RecDB, 
    #            it is because PID doesn't have any records to contribute to that Group.
    return RecName, df_RecDB, df_RecDB_Store


def process_CONFIG_CkpdRecNameFlt_of_PDTInfo(PDTInfo, CONFIG_PDT, CONFIG_CkpdRecNameFlt, df_RecDB_Store, 
                                              UTILS_Flt = UTILS_Flt):
    PDTInfo = PDTInfo.copy()
    CONFIG_Ckpd = CONFIG_CkpdRecNameFlt['CONFIG_Ckpd']
    CONFIG_RecDB = CONFIG_CkpdRecNameFlt['CONFIG_RecDB']
    CONFIG_Flt = CONFIG_CkpdRecNameFlt['CONFIG_Flt']
    
    # -------- initial df_PDT
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    
    # -------- 1. CkpdName
    CkpdName = CONFIG_Ckpd['CkpdName'] # 'Bf1D'
    Ckpd_ARGS = {
        'CkpdName':  CONFIG_Ckpd['CkpdName'], 
        'DistStartToPredDT': CONFIG_Ckpd['DistStartToPredDT'],
        'DistEndToPredDT':  CONFIG_Ckpd['DistEndToPredDT'],
        'TimeUnit':  CONFIG_Ckpd['TimeUnit'], 
    }
    PDTInfo[CkpdName] = get_Ckpd_from_PredDT(PDTInfo['PredDT'], **Ckpd_ARGS)
    
    # -------- 2. CkpdName.RecName
    RecName =  CONFIG_RecDB['RecName']
    RecNameTpt =  CONFIG_RecDB['RecNameTpt']
    df_RecDB = df_RecDB_Store[RecName]
    CkpdRec = CkpdName + '.' + RecName
    # P_RecDB could be empty.
    P_RecDB = df_RecDB[df_RecDB['PID'] == PDTInfo['PID']].iloc[0]
    PDTInfo[CkpdRec] = get_CkpdName_RecTable(PDTInfo, CkpdName, P_RecDB, RecNameTpt)
    
    # -------- 3. CkpdName.RecName.Filter
    FilterName = CONFIG_Flt['FilterName'] # 'Whl'
    CkpdRecFlt = CkpdRec + '.' + FilterName
    
    # if FilterName != 'Whl':
    FilterName2FilterFunc = UTILS_Flt['FilterName2FilterFunc']
    FilterName2Share = UTILS_Flt['FilterName2Share']
    RecName2ThresConfig = UTILS_Flt['RecName2ThresConfig']
    CkpdRecFilter_ARGS = {
        'FilterName': FilterName,
        'FilterFunc': FilterName2FilterFunc[FilterName],
        'ThresShare': FilterName2Share[FilterName],
        'ThresConfig': RecName2ThresConfig[RecNameTpt.split('_')[0]]
    }
    # df_PDT[CkpdRecFlt] = df_PDT.apply(lambda x: filter_CkpdRec_fn(x['PredDT'], x[CkpdRec], **CkpdRecFilter_ARGS), axis = 1)
    PDTInfo[CkpdRecFlt] = filter_CkpdRec_fn(PDTInfo['PredDT'], PDTInfo[CkpdRec], **CkpdRecFilter_ARGS) # PDTInfo[CkpdRec]
    
    PDTInfo_CkpdRecFlt = PDTInfo
    return PDTInfo_CkpdRecFlt


def get_df_TknDB_of_PDTInfo(PDTInfo_CRF, CONFIG_PDT, CONFIG_TknDB, df_TknDB_Store, RANGE_SIZE):

    PDTInfo = PDTInfo_CRF.copy()
    CkpdRecFlt = PDTInfo.keys()[-1]
    Ckpd, RecName, FilterName = CkpdRecFlt.split('.')
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    
    # load variables
    fldgrn_folder = CONFIG_TknDB['fldgrn_folder']
    GrnName = CONFIG_TknDB['GrnName']
    RecFldGrn_List = CONFIG_TknDB['RecFldGrn_List'] 
    SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
    GrnSeqChain_ARGS = {'PID': {'folder': pdt_folder, 'RecName': PDTName}}
    GrnSeqInfo_ARGS = {fld:  {'folder': fldgrn_folder,  'RecName': fld,  'Columns': 'ALL'} for fld in RecFldGrn_List}
    
    # check whether the group information is store in the df_RecBD_Store.
    # Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    df_TknDB_v1 = df_TknDB_Store.get(SynFldGrn, pd.DataFrame(columns = ['PID', 'Group']))
    if Group not in df_TknDB_v1['Group'].unique() : 
        df_TknDB_v2 = get_group_info(Group, GrnSeqChain_ARGS, GrnSeqInfo_ARGS)
        df_TknDB = pd.concat([df_TknDB_v1, df_TknDB_v2]).reset_index(drop = True)
    else:
        df_TknDB = df_TknDB_v1  
        
    df_TknDB_Store[SynFldGrn] = df_TknDB
    return SynFldGrn, df_TknDB, df_TknDB_Store


def process_CONFIG_TknGrn_of_PDTInfoCRF(PDTInfo_CRF, CONFIG_PDT, CONFIG_TknDB, df_TknDB_Store):
    PDTInfo = PDTInfo_CRF.copy()

    # CkpdRecFlt = df_PDT.columns[-1]
    CkpdRecFlt = PDTInfo.keys()[-1]
    
    Ckpd, RecName, FilterName = CkpdRecFlt.split('.')
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    
    # -------- CONFIG_GrnSeq
    fldgrn_folder = CONFIG_TknDB['fldgrn_folder']
    fldgrnv_folder = CONFIG_TknDB['fldgrnv_folder']
    GrnName = CONFIG_TknDB['GrnName']
    RecFldGrn_List = CONFIG_TknDB['RecFldGrn_List'] 
    RDTCal_rfg = CONFIG_TknDB['RDTCal_rfg'] 
    SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
    CkpdRecFltGrn = CkpdRecFlt + '.' + SynFldGrn
    
    # get df ckpdrecfltgrn
    if type(PDTInfo[CkpdRecFlt]) == type(pd.DataFrame()):
        df_TknDB = df_TknDB_Store[SynFldGrn]
        P_TknDB = df_TknDB[df_TknDB['PID'] == PDTInfo['PID']].iloc[0]
        PDTInfo[CkpdRecFltGrn] = get_df_ckpdrecfltgrn(PDTInfo[CkpdRecFlt], P_TknDB, RecFldGrn_List, RDTCal_rfg, RecName, SynFldGrn)
    else:
        PDTInfo[CkpdRecFltGrn] = generate_dfempty(PDTInfo['PID'], PDTInfo['PredDT'], CkpdRecFltGrn, UNK_TOKEN)
    
    PDTInfo_CkpdRecFltTkn = PDTInfo
    # SynFldVocab = build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, UNK_TOKEN)  
    return PDTInfo_CkpdRecFltTkn # , CkpdRecFltTkn, SynFldVocab


def process_CONFIG_CMP_of_PDTInfoCRFT(PDTInfo_CRFT, CONFIG_PDT, CONFIG_TknDB, CONFIG_CMP, 
                                      SynFldVocabNew, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat, 
                                      UTILS_CMP = UTILS_CMP, UNK_TOKEN = UNK_TOKEN):
    # df_PDT = df_PDT_CkpdRecFltTkn.copy()
    PDTInfo = PDTInfo_CRFT.copy()
    
    # CkpdRecFlt = df_PDT.columns[-2]
    CkpdRecFlt = PDTInfo.keys()[-2]
    # CkpdRecFltGrn =  df_PDT.columns[-1]
    CkpdRecFltGrn =  PDTInfo.keys()[-1]
    CkpdName, RecName, FilterName = CkpdRecFlt.split('.')

    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    
    # -------- CONFIG_CMP
    fldgrn_folder = CONFIG_TknDB['fldgrn_folder']
    fldgrnv_folder = CONFIG_TknDB['fldgrnv_folder']
    GrnName = CONFIG_TknDB['GrnName']
    RecFldGrn_List = CONFIG_TknDB['RecFldGrn_List']  
    RDTCal_rfg = CONFIG_TknDB['RDTCal_rfg'] 
    SynFldGrn = f'Rof{RecName}{FilterName}-{GrnName}'
    
    
    method_to_fn = UTILS_CMP['method_to_fn']
    CompressArgs = CONFIG_CMP['CompressArgs']
    prefix_layer_cols = CONFIG_CMP['prefix_layer_cols']
    focal_layer_cols = CONFIG_CMP['focal_layer_cols']
    

    PDTInfo[CkpdRecFltGrnCmp] = compress_dfrec_with_CompressArgs(PDTInfo[CkpdRecFltGrn], CompressArgs, SynFldGrn, 
                                                                 CkpdRecFltGrnCmp, SynFldVocabNew, method_to_fn)
    
    PDTInfo[CkpdRecFltGrnCmpFeat] = convert_df_compressed_to_DPLevel_tensor(PDTInfo[CkpdRecFltGrnCmp], SynFldVocabNew, 
                                                                            CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat, 
                                                                            prefix_layer_cols, focal_layer_cols)
    
    PDTInfo_CkpdRecFltTknCmpFeat = PDTInfo
    return PDTInfo_CkpdRecFltTknCmpFeat # , SynFldVocab, CkpdRecFltGrnCmp, CkpdRecFltGrnCmpFeat