import sys
import open3d
import numpy as np
import time
import os
from geometric_registration.utils import get_pcd, get_keypts, get_desc, loadlog
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def deal_with_one_scene(scene, desc_name, timestr, num_keypts):
    """
    calculate the relative repeatability under {num_keypts} settings for {scene}
    """
    pcdpath = f"../data/3DMatch/fragments/{scene}/"
    keyptspath = f"../geometric_registration/{desc_name}_{timestr}/keypoints/{scene}"
    gtpath = f'../geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)

    # register each pair
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    num_repeat_list = []
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
            if key not in gtLog.keys():
                continue
            source_keypts = get_keypts(keyptspath, cloud_bin_s)
            target_keypts = get_keypts(keyptspath, cloud_bin_t)
            source_keypts = source_keypts[-num_keypts:, :]
            target_keypts = target_keypts[-num_keypts:, :]
            gtTrans = gtLog[key]
            pcd = open3d.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(target_keypts)
            pcd.transform(gtTrans)
            target_keypts = np.asarray(pcd.points)
            distance = cdist(source_keypts, target_keypts, metric='euclidean')
            num_repeat = np.sum(distance.min(axis=0) < 0.1)
            num_repeat_list.append(num_repeat * 1.0 / num_keypts)
    # print(f"Scene {scene} repeatability: {sum(num_repeat_list) / len(num_repeat_list)}")
    return sum(num_repeat_list) / len(num_repeat_list)


def calculate_repeatability(desc_name, timestr, num_keypts):
    """
    calculate the relative repeatability of {desc_name}_{timestr} under {num_keypts} setting.
    """
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    scene_repeatability_list = []
    for scene in scene_list:
        repeatability = deal_with_one_scene(scene, desc_name, timestr, num_keypts=num_keypts)
        scene_repeatability_list.append(repeatability)
    ave_repeatability = sum(scene_repeatability_list) / len(scene_list)
    print(f"Average Repeatability at num_keypts = {num_keypts}: {ave_repeatability}")
    return ave_repeatability


if __name__ == '__main__':
    desc_name = sys.argv[1]
    timestr = sys.argv[2]
    num_list = [4, 8, 16, 32, 64, 128, 256, 512]
    rep_list = []
    for i in num_list:
        ave_repeatability = calculate_repeatability(desc_name, timestr, i)
        rep_list.append(ave_repeatability)
