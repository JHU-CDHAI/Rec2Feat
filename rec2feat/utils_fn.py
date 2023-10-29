######################## Hyper Paramters
FilterName2FilterFunc = {
    'Whl': lambda x: True, 
    'Morning': lambda x: x['DT'].hour in [6,7,8,9,10,11],
    'Afternoon': lambda x: x['DT'].hour in [12, 13, 14, 15, 16, 17],
    'Evening': lambda x: x['DT'].hour in [18, 19, 20, 21, 22, 23],
    'Night': lambda x: x['DT'].hour in [0, 1, 2, 3, 4, 5],
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
    },
    'WeightU': {
        'RInDT': {'Threshold': 0,       'MaxNum': None,}, 
        'DTInD': {'Threshold': 0,       'MaxNum': None, }, 
        'DInCP': {'Threshold': 0,       'MaxNum': None, },
    }
}
########################



UTILS_Flt = {
    'FilterName2FilterFunc': FilterName2FilterFunc,
    'FilterName2Share': FilterName2Share, 
    'RecName2ThresConfig': RecName2ThresConfig, 
}

############ Hyperparameters
method_to_fn = {
    'mean': lambda df: df.mean().round(3),
    'sum': lambda df: df.sum().round(3), 
    'cat': lambda df: df.sum(),
    # TODO: you need to define more compression fn.
}
############


UTILS_CMP = {
    'method_to_fn': method_to_fn, 
}