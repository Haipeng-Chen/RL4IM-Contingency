import os
import sys


def generate_id(path):
    """
    Generate id for training
    
    Parameters
    ----------
        path: str
            model or summary path
    
    Returns
    -------
        id: str
            a identity number
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    dirs = [int(v) for v in os.listdir(path) if len(v) == 3]
    try:
        if len(dirs) == 0:
            return '001'

        new_id = max(dirs) + 1
        if len(str(new_id)) > 3:
            raise ValueError('new id is larger than 999')
        return '0'*(3-len(str(new_id))) + str(new_id)

    except ValueError as e:
        sys.exit(-1)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
