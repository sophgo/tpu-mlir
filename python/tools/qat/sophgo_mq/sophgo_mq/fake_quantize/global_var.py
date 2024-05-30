import torch

def _init():
    global all_data_dict
    all_data_dict = {}

def set_value(key, value):
    all_data_dict[key] = value

def get_value(key):
    try:
        return all_data_dict[key]
    except:
        print('all_data_dict has no', key)

def get_var():
    return all_data_dict