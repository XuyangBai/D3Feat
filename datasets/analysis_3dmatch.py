import os
from os.path import join
import time
import numpy as np
import open3d

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

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
num_test = 0

anc_points = []
for scene in scene_list:

    test_path = '../data/3DMatch/fragments/{}'.format(scene)
    pcd_list = [filename for filename in os.listdir(test_path) if filename.endswith('ply')]
    num_test += len(pcd_list)

    pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
    for i, ind in enumerate(pcd_list):
        pcd = open3d.read_point_cloud(join(test_path, ind))
        pcd = open3d.voxel_down_sample(pcd, voxel_size=0.03)

        # keypts_location = np.fromfile(join(test_path, ind.replace("ply", "keypts.bin")), dtype=np.float32)
        # num_keypts = int(keypts_location[0])
        # keypts_location = keypts_location[1:].reshape([num_keypts, 3])
        # # find the keypoint indices
        # kdtree = open3d.KDTreeFlann(pcd)
        # keypts_id = []
        # for j in range(keypts_location.shape[0]):
        #     _, id, _ = kdtree.search_knn_vector_3d(keypts_location[j], 1)
        #     keypts_id.append(id[0])
        # keypts_id = np.array(keypts_id)

        points = np.array(pcd.points)
        anc_points += [points]

min_list = np.array([pts.min(axis=0) for pts in anc_points])
max_list = np.array([pts.max(axis=0) for pts in anc_points])
mean_list = np.array([pts.mean(axis=0) for pts in anc_points])
length_list = np.array([pts.shape[0] for pts in anc_points])
print("Average Min: {}".format(min_list.mean(axis=0)))
print("Average Max: {}".format(max_list.mean(axis=0)))
print("Average Mean: {}".format(mean_list.mean(axis=0)))
print("Average Pts Size: {}".format(length_list.mean()))
