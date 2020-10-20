"""
---
Frankenstein code
---
Utilities to grab TUdataset from pyTorch Geometric 
https://github.com/rusty1s/pytorch_geometric
"""

from __future__ import print_function
from six.moves import urllib
import os.path as osp
import os
import tarfile
import zipfile
import bz2
import gzip
import shutil
import errno
from distutils.dir_util import copy_tree

URL = 'https://www.chrsmrrs.com/graphkerneldatasets'
cleaned_URL = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')


def get_dataset(dataset, regression=False, multi_target_regression=False):
    print(osp.dirname(osp.realpath(__file__)))
    path = osp.dirname(osp.realpath(__file__))
    download(path,dataset)
    

def download(root,name,cleaned = False):
        url = cleaned_URL if cleaned else URL
        folder = osp.join(root, name)
        path = download_url('{}/{}.zip'.format(url, name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        raw_dir = osp.join(root, name, name)
        copy_tree(raw_dir, folder)
        shutil.rmtree(raw_dir)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)
    data = urllib.request.urlopen(url)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

    

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_tar(path, folder, mode='r:gz', log=True):
    r"""Extracts a tar archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        mode (string, optional): The compression mode. (default: :obj:`"r:gz"`)
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_bz2(path, folder, log=True):
    maybe_log(path, log)
    with bz2.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e