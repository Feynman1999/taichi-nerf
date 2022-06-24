import os
import os.path as osp
import sys
from pathlib import Path
import six
from .misc import is_str

FileNotFoundError = FileNotFoundError

def is_filepath(x):
    if is_str(x) or isinstance(x, Path):
        return True
    else:
        return False

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)
