import pandas as pd


def collate_fn_for_df_PDT(batch_input):
    ##############
    # inputs: you can check the following inputs in the above cells.
    # (1): relational_list
    # (2): new_full_recfldgrn
    # (3): suffix
    ##############
    df_PDT = pd.DataFrame([i for i in batch_input]).reset_index(drop = True)
    return df_PDT