import os
import sys
import numpy as np
import PCLKeypoint

def ensure_keypts_number(pts, keypts, desired_num):
    if keypts.shape[0] == desired_num:
        return keypts
    elif keypts.shape[0] > desired_num:
        return keypts[np.random.choice(keypts.shape[0], desired_num, replace=False)]
    else:
        additional_keypts = pts[np.random.choice(pts.shape[0], desired_num - keypts.shape[0], replace=False)]
        return np.concatenate([keypts, additional_keypts], axis=0).astype(np.float32)

if __name__ == '__main__':
    npy_path = 'numpy_3dmatch_003'
    keypts_parent_path = 'keypoints_baseline'
    detector_name = sys.argv[1]
    num_keypts = int(sys.argv[2])
    keypts_path = os.path.join(keypts_parent_path, detector_name + '_' + str(num_keypts))
    print(f"Keypoints will be saved at {keypts_path}")
    if not os.path.exists(keypts_path):
        os.mkdir(keypts_path)

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
        npy_scene_path = os.path.join(npy_path, scene)
        scene_path = os.path.join(keypts_path, scene)
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        # for each scene, generate the keypoints and save to keypoints_baseline/{detector_name}_{num_keypts}
        filelist = os.listdir(npy_scene_path)
        for filename in filelist:
            pts = np.load(os.path.join(npy_scene_path, filename)) 
            if detector_name == 'random':
                keypts = pts[np.random.choice(len(pts), num_keypts)].astype(np.float32)
                keypts.tofile(os.path.join(scene_path, filename).replace("npy", "bin").replace("cloud_bin_", ""))
            elif detector_name == 'sift':
                num_to_contrast = {
                    4: 0.03, 
                    8: 0.025, 
                    16: 0.015, 
                    32: 0.01, 
                    64: 0.0075, 
                    128: 0.005,
                    256: 0.004,
                    512: 0.0025
                }
                keypts = PCLKeypoint.keypointSift(pts,                                                           
                                                min_scale=0.05,
                                                n_octaves=4,
                                                n_scales_per_octave=8,
                                                min_contrast=num_to_contrast[num_keypts])
                print(keypts.shape)
                keypts = ensure_keypts_number(pts, keypts, num_keypts)
                keypts.tofile(os.path.join(scene_path, filename).replace("npy", "bin").replace("cloud_bin_", ""))
                # np.save(os.path.join(scene_path, filename), keypts)
            elif detector_name == 'harris':
                num_to_radius = {
                    4: 0.4, 
                    8: 0.3, 
                    16: 0.2, 
                    32: 0.15, 
                    64: 0.1, 
                    128: 0.05,
                    256: 0.05,
                    512: 0.05
                }
                num_to_nms = {
                    4: 0.001, 
                    8: 0.001, 
                    16: 0.001, 
                    32: 0.001, 
                    64: 0.001, 
                    128: 0.001,
                    256: 0.00001,
                    512: 1e-8
                }
                keypts = PCLKeypoint.keypointHarris(pts,                                                           
                                                radius=num_to_radius[num_keypts],
                                                nms_threshold=num_to_nms[num_keypts],
                                                threads=0)
                print(keypts.shape)
                keypts = ensure_keypts_number(pts, keypts, num_keypts)
                keypts.tofile(os.path.join(scene_path, filename).replace("npy", "bin").replace("cloud_bin_", ""))
            elif detector_name == 'iss':
                num_to_radius = {
                    4: 0.6,
                    8: 0.5, 
                    16: 0.4, 
                    32: 0.3, 
                    64: 0.2, 
                    128: 0.15,
                    256: 0.10,
                    512: 0.05
                }
                keypts = PCLKeypoint.keypointIss(pts,
                                                iss_salient_radius=num_to_radius[num_keypts],
                                                iss_non_max_radius=num_to_radius[num_keypts],
                                                iss_gamma_21=0.975,
                                                iss_gamma_32=0.975,
                                                iss_min_neighbors=5,
                                                threads=0)
                print(keypts.shape)
                keypts = ensure_keypts_number(pts, keypts, num_keypts)
                keypts.tofile(os.path.join(scene_path, filename).replace("npy", "bin").replace("cloud_bin_", ""))
            else:
                print("No such detector")
