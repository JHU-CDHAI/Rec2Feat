import pandas as pd
from recfldgrn.datapoint import convert_PID_to_PIDgroup
from .ckpdrec import get_group_info
from .ckpdrecflt import add_RelativeDT_to_dfCkpdRec

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
    PredDT = df['PredDT'].iloc[0]
    df = add_RelativeDT_to_dfCkpdRec(df, PredDT)
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
    # print(DTRel_cols, df.columns)
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

def get_df_TknDB_of_PDTInfo(Case_CRF, SynFldGrn, CONFIG_PDT, CONFIG_TknDB, df_TknDB_Store, RANGE_SIZE):

    PDTInfo = Case_CRF# .copy()
    pdt_folder = CONFIG_PDT['pdt_folder']
    PDTName = CONFIG_PDT['PDTName']
    Group = convert_PID_to_PIDgroup(PDTInfo['PID'], RANGE_SIZE)
    
    # load variables
    fldgrn_folder = CONFIG_TknDB['fldgrn_folder']
    RecFldGrn_List = CONFIG_TknDB['RecFldGrn_List'] 
    GrnSeqChain_ARGS = {'PID': {'folder': pdt_folder, 'RecName': PDTName}}
    GrnSeqInfo_ARGS = {fld:  {'folder': fldgrn_folder,  'RecName': fld,  'Columns': 'ALL'} for fld in RecFldGrn_List}
    
    # check whether the group information is store in the df_RecBD_Store.
    df_TknDB_v1 = df_TknDB_Store.get(SynFldGrn, pd.DataFrame(columns = ['PID', 'Group']))
    if Group not in df_TknDB_v1['Group'].unique() : 
        df_TknDB_v2 = get_group_info(Group, GrnSeqChain_ARGS, GrnSeqInfo_ARGS)
        df_TknDB = pd.concat([df_TknDB_v1, df_TknDB_v2]).reset_index(drop = True)
    else:
        df_TknDB = df_TknDB_v1  
        
    df_TknDB_Store[SynFldGrn] = df_TknDB
    return SynFldGrn, df_TknDB, df_TknDB_Store


def process_CONFIG_TknDB_of_PDTInfoCRF(Case_CRF, CkpdRecFltTkn, CONFIG_TknDB, df_TknDB_Store, UNK_TOKEN):
    PDTInfo = Case_CRF# .copy()
    Ckpd, RecName, FilterName, SynFldGrn = CkpdRecFltTkn.split('.')
    CkpdRecFlt = '.'.join([Ckpd, RecName, FilterName])

    RecFldGrn_List = CONFIG_TknDB['RecFldGrn_List'] 
    RDTCal_rfg = CONFIG_TknDB['RDTCal_rfg'] 

    if type(PDTInfo[CkpdRecFlt]) == type(pd.DataFrame()):
        df_TknDB = df_TknDB_Store[SynFldGrn]
        P_TknDB = df_TknDB[df_TknDB['PID'] == PDTInfo['PID']].iloc[0]
        CRFTCValue = get_df_ckpdrecfltgrn(PDTInfo[CkpdRecFlt], P_TknDB, RecFldGrn_List, RDTCal_rfg, RecName, SynFldGrn)
    else:
        CRFTCValue = generate_dfempty(PDTInfo['PID'], PDTInfo['PredDT'], CkpdRecFltTkn, UNK_TOKEN)

    # PDTInfo[CkpdRecFltTkn] = CRFTCValue.drop(columns = ['DT', 'R'])
    PDTInfo[CkpdRecFltTkn] = CRFTCValue
    Case_CRFT = PDTInfo
    
    return Case_CRFT 

