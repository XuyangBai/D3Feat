import open3d
import os
import pickle
import numpy as np

# because PCLKeypoint and open3d are incompatible

def process_3dmatch():
    origin_ply_path = 'data/3DMatch/fragments/'
    save_npy_path = 'repeatability/numpy_003'
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

    for scene in scene_list:
        if not os.path.exists(os.path.join(save_npy_path, scene)):
            os.makedirs(os.path.join(save_npy_path, scene))
        ply_path = os.path.join(origin_ply_path, scene)
        num_ply = len([filename for filename in os.listdir(ply_path) if filename.endswith('ply')])
        for i in range(num_ply):
            filepath = os.path.join(origin_ply_path, scene, f'cloud_bin_{i}.ply')
            pcd = open3d.read_point_cloud(filepath)
            pcd = open3d.voxel_down_sample(pcd, voxel_size=0.03)
            pts = np.asarray(pcd.points)
            np.save(os.path.join(save_npy_path, scene, f'cloud_bin_{i}.npy'), pts)

def process_kitti():
    save_npy_path = 'repeatability/numpy_kitti_030'
    from datasets.KITTI import KITTIDataset
    dataset = KITTIDataset(1, first_subsampling_dl=0.3, load_test=True)
    gtTrans = {}
    for i in range(len(dataset.files['test'])):
        (_, _, unaligned_anc_points, unaligned_pos_points, matches, trans, flag) = dataset.__getitem__('test', i)
        if flag == False:
            continue
        drive = dataset.files['test'][i][0]
        t0, t1 =  dataset.files['test'][i][1], dataset.files['test'][i][2]

        np.save(os.path.join(save_npy_path, f'{drive}@{t0}.npy'), unaligned_anc_points)
        np.save(os.path.join(save_npy_path, f'{drive}@{t1}.npy'), unaligned_pos_points)
        gtTrans[dataset.files['test'][i]] = trans
    with open(os.path.join(save_npy_path, 'gt.pkl'), 'wb') as f:
        pickle.dump(gtTrans, f)

if __name__ == '__main__':
    # process_3dmatch()
    process_kitti()
