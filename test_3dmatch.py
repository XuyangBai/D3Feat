# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.tester import ModelTester
from models.KPFCNN_model import KernelPointFCNN

# Datasets
from datasets.ThreeDMatch import ThreeDMatchDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def test_caller(path, step_ind, on_val):

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_color = 1.0
    #config.validation_size = 500
    #config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    dataset = ThreeDMatchDataset(1, load_test=True)

    # Initialize input pipelines
    dataset.init_test_input_pipeline(config)


    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    tester.generate_descriptor(model, dataset)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    # Default is last log and last snapshot
    chosen_log = 'last_3DMatch'
    chosen_snapshot = -1
    on_val = False

    ###########################
    # Call the test initializer
    ###########################

    # Dataset name
    test_dataset = '3DMatch'

    # List all training logs
    logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
    # Find the last log of asked dataset or manually specify the log path
    for log in logs[::-1]:
        log_config = Config()
        log_config.load(log)
        if log_config.dataset.startswith(test_dataset):
            chosen_log = log
            break
    # chosen_log = `results/Log_`

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot, on_val)



