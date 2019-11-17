import os
from os.path import join
import time 
import numpy as np
import open3d
open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

scene_list = [
    'gazebo_summer',
    'gazebo_winter',
    'wood_autmn',
    'wood_summer',
]
anc_points = []
for scene in scene_list:
    test_path = '../data/ETH/{}'.format(scene)
    pcd_list = [filename for filename in os.listdir(test_path) if filename.endswith('ply')]

    pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
    for i, ind in enumerate(pcd_list):
        pcd = open3d.read_point_cloud(join(test_path, ind))
        pcd = open3d.voxel_down_sample(pcd, voxel_size=0.03)

        points = np.asarray(pcd.points)
        anc_points += [points]
        
min_list = np.array([pts.min(axis=0) for pts in anc_points])
max_list = np.array([pts.max(axis=0) for pts in anc_points])
mean_list = np.array([pts.mean(axis=0) for pts in anc_points])
length_list = np.array([pts.shape[0] for pts in anc_points])
print("{0} Average Min: {1}".format(scene, min_list.mean(axis=0)))
print("{0} Average Max: {1}".format(scene, max_list.mean(axis=0)))
print("{0} Average Mean: {1}".format(scene, mean_list.mean(axis=0)))
print("{0} Average Pts Size: {1}".format(scene, length_list.mean()))