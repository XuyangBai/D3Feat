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


if __name__  == '__main__':
    npy_path = 'numpy_kitti_030'
    keypts_parent_path = 'keypoints_baseline_kitti'
    detector_name = sys.argv[1]
    num_keypts = int(sys.argv[2])
    keypts_path = os.path.join(keypts_parent_path, detector_name + '_' + str(num_keypts))
    print(f"Keypoints will be saved at {keypts_path}")
    if not os.path.exists(keypts_path):
        os.mkdir(keypts_path)

    for filename in os.listdir(npy_path):
        if not filename.endswith('.npy'):
            continue
        pts = np.load(os.path.join(npy_path, filename))
        if detector_name == 'random':
            keypts = pts[np.random.choice(len(pts), num_keypts)].astype(np.float32)
            keypts.tofile(os.path.join(keypts_path, filename).replace("npy", "bin").replace("cloud_bin_", ""))
        elif detector_name == 'sift':
            keypts = PCLKeypoint.keypointSift(pts,                                                           
                                            min_scale=0.5,
                                            n_octaves=4,
                                            n_scales_per_octave=8,
                                            min_contrast=0.1)
            print(keypts.shape)
            keypts = ensure_keypts_number(pts, keypts, num_keypts)
            keypts.tofile(os.path.join(keypts_path, filename).replace("npy", "bin"))
        elif detector_name == 'harris':
            keypts = PCLKeypoint.keypointHarris(pts,                                                           
                                            radius=1,
                                            nms_threshold=0.001,
                                            threads=0)
            print(keypts.shape)
            keypts = ensure_keypts_number(pts, keypts, num_keypts)
            keypts.tofile(os.path.join(keypts_path, filename).replace("npy", "bin"))
        elif detector_name == 'iss':
            keypts = PCLKeypoint.keypointIss(pts,
                                            iss_salient_radius=2,
                                            iss_non_max_radius=2,
                                            iss_gamma_21=0.975,
                                            iss_gamma_32=0.975,
                                            iss_min_neighbors=5,
                                            threads=0)
            print(keypts.shape)
            keypts = ensure_keypts_number(pts, keypts, num_keypts)
            keypts.tofile(os.path.join(keypts_path, filename).replace("npy", "bin"))
        else:
            print("No such detector")

        

        
        