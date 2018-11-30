from keras import backend as K
import os
import tensorflow as tf
from src.models import *
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def train_model(exp_type, num_trials, index, hyperparams,
            n_train=100000, n_val=10000, n_test=10000, seed=821):

    sess = tf.Session()
    K.set_session(sess)

    if not os.path.exists('log/' + exp_type):
        os.makedirs('log/' + exp_type)

    data_type = int(hyperparams['data_type'])
    img_size = int(hyperparams['img_size'])
    input_depth = int(hyperparams['img_depth'])
    forecasting_horizon = int(hyperparams['forecasting_horizon'])
    seq_len = int(hyperparams['seq_len'])
    model_type = str(hyperparams['model_type'])

    batch_size = int(hyperparams['batch_size'])
    max_epoch = int(hyperparams['max_epoch'])
    
    optimizer = str(hyperparams['optimizer'])
    opt_learningrate = float(hyperparams['opt_learningrate'])
    batchnorm_on = int(hyperparams['batchnorm_on']) #0 or 1
    dropout_on = int(hyperparams['dropout_on']) #0 or 1
    loss_func = str(hyperparams['loss_func'])
    earlystop_on = int(hyperparams['earlystop_on']) #0 or 1

    n_conv_layers = int(hyperparams['n_conv_layers'])
    conv1_depth = int(hyperparams['conv1_depth'])
    conv2_depth = int(hyperparams['conv2_depth'])
    conv3_depth = int(hyperparams['conv3_depth'])
    conv4_depth = int(hyperparams['conv4_depth'])
    conv5_depth = int(hyperparams['conv5_depth'])
    conv_fc_units = int(hyperparams['conv_fc_units'])
    act_conv = str(hyperparams['act_conv'])
    pooling_on = int(hyperparams['pooling_on'])
    pooling_size = int(hyperparams['pooling_size'])
    conv_filter_size = int(hyperparams['conv_filter_size'])

    n_lstm_layers = int(hyperparams['n_lstm_layers'])
    lstm_units = int(hyperparams['lstm_units'])
    act_lstm = str(hyperparams['act_lstm'])
    lstm_fc_units = int(hyperparams['lstm_fc_units'])

    n_fc_layers = int(hyperparams['n_fc_layers'])
    last_fc_units = int(hyperparams['last_fc_units'])

    temp_type = str(hyperparams['temp_type'])
    traj_opt = str(hyperparams['traj_opt'])
    normalization_opt = str(hyperparams['normalization_opt'])

    if normalization_opt == 'raw':
        spdDict = np.load('src/preprocessing/data/spd_interpolated_ignoreErrTimes.npy').item()
        spdArray = np.load('src/preprocessing/data/spdArray.npy')
    else:
        spdDict = np.load('src/preprocessing/data/spdDict_spdLimitNorm_' + str(normalization_opt) + '.npy').item()
        spdArray = np.load('src/preprocessing/data/spdArray_spdLimitNorm_' + str(normalization_opt) + '.npy')

    for trial_no in range(num_trials):
        no_exp = exp_type + '/' + str(forecasting_horizon) + 'hours_dataType' + str(data_type) + '_normalization' + str(normalization_opt) + '_traj' + str(traj_opt) + \
                    '_img' + str(img_size) + '_seq' + str(seq_len) + '_trial' + str(trial_no)

        print('-----------------------------------------------------------------------------')
        print(no_exp)
        print(str(float(index) + 1) + ' / ' + str(float(models.shape[0])) + ' / trial : ' + str(trial_no))
        print('data_type: %i, seq_len: %i, img_size: %i, normalization: %s, traj: %s' %(data_type, seq_len, img_size, normalization_opt, traj_opt))
        print('-----------------------------------------------------------------------------')

        if model_type == 'lstm':
            lstm_model(no_exp, data_type, forecasting_horizon, seq_len, spdDict,
                        n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
                        opt_learningrate, loss_func, max_epoch,
                        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed, 
                        temp_type=temp_type, normalization_opt=normalization_opt)

        elif model_type == 'cnnlstm':
            cnnlstm_model(no_exp, data_type, img_size, input_depth, forecasting_horizon, seq_len, spdArray,
                            batch_size, batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth,
                            conv3_depth, conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                            n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                            n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,  
                            traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)

        elif model_type == 'ndlstm':
            nd_lstm_model(no_exp, data_type, img_size, spdDict, forecasting_horizon, seq_len, spdArray, batch_size, dropout_on,
                            n_lstm_layers, lstm_units, act_lstm, lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                            n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,  
                            traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)

        elif model_type == '3dcnn':
            cnn3d_model(no_exp, data_type, img_size, forecasting_horizon, seq_len, spdArray, batch_size,
                        batchnorm_on, dropout_on, n_conv_layers, conv1_depth, conv2_depth, conv3_depth,
                        conv4_depth, conv5_depth, conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
                        lstm_fc_units, n_fc_layers, last_fc_units, opt_learningrate, loss_func, max_epoch,
                        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,  
                        traj_opt=traj_opt, temp_type=temp_type, normalization_opt=normalization_opt)


def main(exp_name):
    
    num_trials = 1
    # read hyperparamters from CSV
    models = pd.read_csv('model/' + str(exp_name) + '.csv')

    for index, hpv in models.iterrows():
        train_model('test_datasize', num_trials, index, hpv)


if __name__ == '__main__':

    exp_name = 'test_datasize'
    gpu_id = 1
    if not os.path.exists('log/' + exp_name):
        os.makedirs('log/' + exp_name)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    main(exp_name, gpu_id)
