import sys
import open3d
import numpy as np
import time
import os
from geometric_registration.utils import get_pcd, get_keypts, get_desc, loadlog
from geometric_registration.evaluate import build_correspondence
from functools import partial


def register2Fragments(id1, id2, keyptspath, descpath, resultpath, gtLog, desc_name, inlier_ratio, distance_threshold):
    """
    Register point cloud {id1} and {id2} using the keypts location and descriptors.
    """
    # cloud_bin_s = f'Hokuyo_{id1}'
    # cloud_bin_t = f'Hokuyo_{id2}'
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        return 0, 0, 0
    source_keypts = get_keypts(keyptspath, cloud_bin_s)
    target_keypts = get_keypts(keyptspath, cloud_bin_t)
    source_desc = get_desc(descpath, cloud_bin_s, desc_name)
    target_desc = get_desc(descpath, cloud_bin_t, desc_name)
    source_desc = np.nan_to_num(source_desc)
    target_desc = np.nan_to_num(target_desc)
    # Select {num_keypts} points based on the scores. The descriptors and keypts are already sorted based on the detection score.
    num_keypts = 250
    source_keypts = source_keypts[-num_keypts:, :]
    source_desc = source_desc[-num_keypts:, :]
    target_keypts = target_keypts[-num_keypts:, :]
    target_desc = target_desc[-num_keypts:, :]
    # Select {num_keypts} points randomly.
    # num_keypts = 250
    # source_indices = np.random.choice(range(source_keypts.shape[0]), num_keypts)
    # target_indices = np.random.choice(range(target_keypts.shape[0]), num_keypts)
    # source_keypts = source_keypts[source_indices, :]
    # source_desc = source_desc[source_indices, :]
    # target_keypts = target_keypts[target_indices, :]
    # target_desc = target_desc[target_indices, :]
    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        # skip the pairs that have less than 30% overlap.
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        # find mutually cloest point.
        corr = build_correspondence(source_desc, target_desc)

        gtTrans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < distance_threshold)
        if num_inliers / len(distance) < inlier_ratio:
            print(key)
            print("num_corr:", len(corr), "inlier_ratio:", num_inliers / len(distance))
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

    # write the result into resultpath so that it can be re-shown.
    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag


def read_register_result(resultpath, id1, id2):
    """
    Read the registration result of {id1} & {id2} from the resultpath
    Return values contain the inlier_number, inlier_ratio, flag(indicating whether this pair is a ground truth match).
    """
    # cloud_bin_s = f'Hokuyo_{id1}'
    # cloud_bin_t = f'Hokuyo_{id2}'
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


def deal_with_one_scene(inlier_ratio, distance_threshold, scene):
    """
    Function to register all the fragments pairs in one scene.
    """
    pcdpath = f"../data/ETH/{scene}/"
    keyptspath = f"{desc_name}_{timestr}/keypoints/{scene}"
    descpath = f"{desc_name}_{timestr}/descriptors/{scene}"
    gtpath = f'../data/ETH/{scene}'
    gtLog = loadlog(gtpath)
    resultpath = f"pred_result/{scene}/{desc_name}_result_{timestr}"
    if not os.path.exists(f"pred_result/{scene}/"):
        os.mkdir(f"pred_result/{scene}/")
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    # register each pair
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    print(f"Start Evaluate Descriptor {desc_name} for {scene}")
    start_time = time.time()
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            register2Fragments(id1, id2, keyptspath, descpath, resultpath, gtLog, desc_name, inlier_ratio, distance_threshold)
    print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    scene_list = [
        'gazebo_summer',
        'gazebo_winter',
        'wood_autmn',
        'wood_summer',
    ]
    # will evaluate the descriptor in `{desc_name}_{timestr}` folder.
    desc_name = sys.argv[1]
    timestr = sys.argv[2]
    inlier_ratio = 0.05
    distance_threshold = 0.10

    # multiprocessing to register each pair in each scene.
    # this part is time-consuming
    from multiprocessing import Pool

    pool = Pool(len(scene_list))
    func = partial(deal_with_one_scene, inlier_ratio, distance_threshold)
    pool.map(func, scene_list)
    pool.close()
    pool.join()

    # collect all the data and print the results.
    inliers_list = []
    recall_list = []
    inliers_ratio_list = []
    pred_match = 0
    gt_match = 0
    for scene in scene_list:
        # evaluate
        pcdpath = f"../data/ETH/{scene}/"
        resultpath = f"pred_result/{scene}/{desc_name}_result_{timestr}"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(resultpath, id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])
        result = np.array(result)
        gt_results = np.sum(result[:, 2] == 1)
        pred_results = np.sum(result[:, 1] > inlier_ratio)
        pred_match += pred_results
        gt_match += gt_results
        recall = float(pred_results / gt_results) * 100
        print(f"Correct Match {pred_results}, ground truth Match {gt_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 2] == 1, result[:, 0], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliners: {ave_num_inliers}")
        ave_inlier_ratio = np.sum(np.where(result[:, 2] == 1, result[:, 1], np.zeros(result.shape[0]))) / pred_results
        print(f"Average Num Inliner Ratio: {ave_inlier_ratio}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
        inliers_ratio_list.append(ave_inlier_ratio)

    print("*" * 40)
    print(recall_list)
    # print(f"True Avarage Recall: {pred_match / gt_match * 100}%")
    print(f"Avergae Matching Recall: {pred_match / gt_match * 100}%")
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
    average_inliers_ratio = sum(inliers_ratio_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers ratio: {average_inliers_ratio}")
