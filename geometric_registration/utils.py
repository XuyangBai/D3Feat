import os
import open3d
import numpy as np


def get_pcd(pcdpath, filename):
    return open3d.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_keypts(keyptspath, filename):
    keypts = np.load(os.path.join(keyptspath, filename + f'.npy'))
    return keypts


def get_desc(descpath, filename, desc_name):
    desc = np.load(os.path.join(descpath, filename + f'.{desc_name}.npy'))
    return desc


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result
