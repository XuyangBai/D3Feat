import sys
import os
import open3d
import numpy as np
import time
import os
from scipy.spatial.distance import cdist

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)


def deal_with_one_pair(source_keypts, target_keypts, trans, num_keypts, threshold):
    """
    calculate the relative repeatability under {num_keypts} settings for one pair.
    """
    gtTrans = trans
    pcd = open3d.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(source_keypts)
    pcd.transform(gtTrans)
    source_keypts = np.asarray(pcd.points)
    distance = cdist(source_keypts, target_keypts, metric='euclidean')
    num_repeat = np.sum(distance.min(axis=0) < threshold)
    return num_repeat * 1.0 / num_keypts


def calculate_repeatability(desc_name, timestr, num_keypts):
    """
    calculate the relative repeatability of {desc_name}_{timestr} under {num_keypts} setting.
    """
    from datasets.KITTI import KITTIDataset
    dataset = KITTIDataset(1, first_subsampling_dl=0.3, load_test=True)
    repeat_list = []
    keyptspath = f"geometric_registration_kitti/{desc_name}_{timestr}"
    for i in range(len(dataset.files['test'])):
        drive = dataset.files['test'][i][0]
        t0, t1 = dataset.files['test'][i][1], dataset.files['test'][i][2]
        filename = f'{drive}@{t0}-{t1}.npz'
        if not os.path.exists(os.path.join(keyptspath, filename)):
            continue
        data = np.load(os.path.join(keyptspath, filename))
        source_keypts = data['anc_pts'][-num_keypts:]
        target_keypts = data['pos_pts'][-num_keypts:]
        repeat_list += [deal_with_one_pair(source_keypts, target_keypts, data['trans'], num_keypts, threshold=0.5)]
    print(f"Average Repeatability at num_keypts = {num_keypts}: {np.mean(repeat_list)}")
    return np.mean(repeat_list)


if __name__ == '__main__':
    desc_name = sys.argv[1]
    timestr = sys.argv[2]

    num_list = [4, 8, 16, 32, 64, 128, 256, 512]
    rep_list = []
    os.chdir("../")
    from multiprocessing import Pool

    # pool = Pool(len(num_list))
    # pool.map(calculate_repeatability, num_list)
    # pool.close()
    # pool.join()

    for i in num_list:
        ave_repeatability = calculate_repeatability(desc_name, timestr, i)
        rep_list.append(ave_repeatability)
