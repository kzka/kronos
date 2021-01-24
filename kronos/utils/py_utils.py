import numpy as np


def dict_mean(dict_list):
    """Take the mean of a list of dicts.

    The dicts must have the same keys.
    """
    # ensure all dicts have the same keys
    keys = dict_list[0].keys()
    for d in dict_list:
        if d.keys() != keys:
            raise ValueError("Dictionary keys must be identical.")

    # combine list of dictionaries into one dictionary where
    # each key's value is a list containing the elements from
    # all the subdictionaries
    res = {k: [dict_list[i][k] for i in range(len(dict_list))] for k in keys}

    # now take the average over the lists
    return {k: np.mean(v) for k, v in res.items()}
