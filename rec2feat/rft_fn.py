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
    
    df_RecDB_v1 = df_RecDB_Store.get(RecName, pd.DataFrame(columns = ['PID']))
    if PDTInfo['PID'] not in df_RecDB_v1['PID'].to_list() : 
        df_RecDB_v2 = get_info_from_settings(Group, None, RecChain_ARGS, RecInfo_ARGS)
        df_RecDB = pd.concat([df_RecDB_v1, df_RecDB_v2]).reset_index(drop = True)
    else:
        df_RecDB = df_RecDB_v1  
        
    df_RecDB_Store[RecName] = df_RecDB
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
    P_RecDB = df_RecDB[df_RecDB['PID'] == PDTInfo['PID']].iloc[0]
    PDTInfo[CkpdRec] = get_CkpdName_RecTable(PDTInfo, CkpdName, P_RecDB, RecNameTpt)
    
    # -------- 3. CkpdName.RecName.Filter
    FilterName = CONFIG_Flt['FilterName'] # 'Whl'
    CkpdRecFlt = CkpdRec + '.' + FilterName
    if FilterName != 'Whl':
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
    else:
        # df_PDT[CkpdRecFlt] = df_PDT[CkpdRec]
        PDTInfo[CkpdRecFlt] = PDTInfo[CkpdRec]
        
    PDTInfo_CkpdRecFlt = PDTInfo
    return PDTInfo_CkpdRecFlt
