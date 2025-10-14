import pickle
import os
def save_dict(mydict, file_path):
    f = open(file_path,"wb")
    pickle.dump(mydict,f)
    
def load_dict(file_path):
    with open(file_path,"rb") as f:
        data = pickle.load(f)
    return data


def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1
