import sys
import open3d
open3d.set_verbosity_level(open3d.VerbosityLevel.Error)
import numpy as np
import time
import os
import pickle
from geometric_registration.utils import get_pcd, get_keypts, get_desc, loadlog
# from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


def deal_with_one_pair(detector_name, triplet, trans, num_keypts, threshold):
    keypts_parent_path = 'repeatability/keypoints_baseline_kitti'
    drive = triplet[0]
    t0,  t1 = triplet[1], triplet[2]
    if detector_name == 'USIP':
        source_keypts = np.fromfile(os.path.join(keypts_parent_path, f'{detector_name}_{num_keypts}/{drive:02d}/{t0:06d}.bin'), dtype=np.float32).reshape([-1, 3])
        target_keypts = np.fromfile(os.path.join(keypts_parent_path, f'{detector_name}_{num_keypts}/{drive:02d}/{t1:06d}.bin'), dtype=np.float32).reshape([-1, 3])     
    else:
        source_keypts = np.fromfile(os.path.join(keypts_parent_path, f'{detector_name}_{num_keypts}/{drive}@{t0}.bin'), dtype=np.float32).reshape([-1, 3])
        target_keypts = np.fromfile(os.path.join(keypts_parent_path, f'{detector_name}_{num_keypts}/{drive}@{t1}.bin'), dtype=np.float32).reshape([-1, 3])   
    gtTrans = trans
    pcd = open3d.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(source_keypts)
    pcd.transform(gtTrans)
    source_keypts = np.asarray(pcd.points)
    distance = cdist(source_keypts, target_keypts, metric='euclidean')
    num_repeat = np.sum(distance.min(axis=0) < threshold)
    return num_repeat * 1.0 / num_keypts

def calculate_repeatability(num_keypts):
    from datasets.KITTI import KITTIDataset
    dataset = KITTIDataset(1, first_subsampling_dl=0.3, load_test=True)
    repeat_list = []
    for i in range(len(dataset.files['test'])):
        # (_, _, unaligned_anc_points, unaligned_pos_points, matches, trans, flag) = dataset.__getitem__('test', i)
        # if flag == False:
            # continue
        trans = gtLog[dataset.files['test'][i]]
        repeat_list += [deal_with_one_pair(detector_name, dataset.files['test'][i], trans, num_keypts, threshold=0.5)]
    print(f"Average Repeatability at num_keypts = {num_keypts}: {np.mean(repeat_list)}")
    return np.mean(repeat_list)

if __name__ == '__main__':
    detector_name = sys.argv[1]
    num_list = [4, 8, 16, 32, 64, 128, 256, 512]
    rep_list = []
    with open('repeatability/numpy_kitti_030/gt.pkl', 'rb') as f:
        gtLog = pickle.load(f)

    for i in num_list:
        ave_repeatability = calculate_repeatability(i)
        rep_list.append(ave_repeatability)
    
    