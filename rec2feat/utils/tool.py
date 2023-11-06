
def whl_filter(x): return True
def morning_filter(x): return x['DT'].hour in [6,7,8,9,10,11]
def afternoon_filter(x): return x['DT'].hour in [12, 13, 14, 15, 16, 17]
def evening_filter(x): return x['DT'].hour in [18, 19, 20, 21, 22, 23]
def night_filter(x): return x['DT'].hour in [0, 1, 2, 3, 4, 5]

#############################
FilterName2FilterFunc = {
    'Whl': whl_filter,
    'Morning':morning_filter,
    'Afternoon': morning_filter, #  lambda x: x['DT'].hour in [12, 13, 14, 15, 16, 17],
    'Evening': afternoon_filter,  # lambda x: x['DT'].hour in [18, 19, 20, 21, 22, 23],
    'Night': night_filter, # lambda x: x['DT'].hour in [0, 1, 2, 3, 4, 5],
}

FilterName2Share = {
    'Whl': 1,
    'Morning': 0.25,
    'Afternoon': 0.25,
    'Evening': 0.25,
    'Night': 0.25,
}

RecName2ThresConfig = {
    'CGM5Min': {
        'RInDT': {'Threshold': 0,       'MaxNum': None,},
        'DTInD': {'Threshold': 288*0.7, 'MaxNum': 288, },
        'DInCP': {'Threshold': 0,       'MaxNum': None, },
    }
}
#############################


UTILS_Flt = {
    'FilterName2FilterFunc': FilterName2FilterFunc,
    'FilterName2Share': FilterName2Share, 
    'RecName2ThresConfig': RecName2ThresConfig, 
}


def df_mean(df): return df.mean().round(3)
def df_sum(df): return df.sum().round(3)
def df_cat(df): return df.sum()

############ Hyperparameters
method_to_fn = {
    'mean': df_mean,  # lambda df: ,
    'sum': df_sum, # lambda df: df.sum().round(3), 
    'cat': df_cat, # lambda df: df.sum(),
    # TODO: you need to define more compression fn.
}
############


UTILS_CMP = {
    'method_to_fn': method_to_fn, 
}


