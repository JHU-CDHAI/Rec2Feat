import os
import pandas as pd
from functools import reduce

def build_SynFldVocab(fldgrnv_folder, RecFldGrn_List, RDTCal_rfg, UNK_TOKEN):
    RecFldGrn_List_new = RecFldGrn_List + [RDTCal_rfg] if type(RDTCal_rfg) == str else RecFldGrn_List
    SynFld_idx2grn = [UNK_TOKEN]
    SynFld_grn2rfg = {UNK_TOKEN:UNK_TOKEN}
    for RecFldGrn in RecFldGrn_List_new:
        RecFldGrnVocab = pd.read_pickle(os.path.join(fldgrnv_folder, RecFldGrn + '.p'))
        SynFld_idx2grn = SynFld_idx2grn + [i for i in RecFldGrnVocab['idx2grn'] if i not in SynFld_idx2grn]
        SynFld_grn2rfg = dict(**SynFld_grn2rfg, **RecFldGrnVocab['grn2rfg'])
    SynFld_grn2idx = {v:k for k, v in enumerate(SynFld_idx2grn)}
    SynFldVocab = pd.Series({'idx2grn': SynFld_idx2grn, 'grn2idx': SynFld_grn2idx, 'grn2rfg':SynFld_grn2rfg} )
    idx2rfg = sorted(list(set(SynFldVocab['grn2rfg'].values())))
    rfg2idx = {v:k for k, v in enumerate(idx2rfg)}
    SynFldVocab['idx2rfg'] = idx2rfg
    SynFldVocab['rfg2idx'] = rfg2idx
    return SynFldVocab

def update_SynFldVocab(SynFldVocab, rfg, grn_list):
    idx2grn = SynFldVocab['idx2grn']
    grn2rfg = SynFldVocab['grn2rfg']
    idx2grn = idx2grn + [i for i in grn_list if i not in idx2grn]
    grn2idx = {v:k for k, v in enumerate(idx2grn)}
    grn2rfg = {**grn2rfg, **{k:rfg for k in grn_list}}
    SynFldVocab['idx2grn'] = idx2grn
    SynFldVocab['grn2idx'] = grn2idx
    SynFldVocab['grn2rfg'] = grn2rfg
    
    idx2rfg = sorted(list(set(SynFldVocab['grn2rfg'].values())))
    rfg2idx = {v:k for k, v in enumerate(idx2rfg)}
    SynFldVocab['idx2rfg'] = idx2rfg
    SynFldVocab['rfg2idx'] = rfg2idx
    return SynFldVocab        

def get_update_SynFldVocab(SynFldVocab, CompressArgs):
    # part 1: update SynFldVocab
    SynFldVocabNew = SynFldVocab.copy()
    for sourceInTarget in CompressArgs:
        SynFldVocabNew = update_SynFldVocab(SynFldVocabNew, sourceInTarget, [sourceInTarget])
    return SynFldVocabNew

