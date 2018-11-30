from wot.interface import *
from main import train_model

@eval_task
def objective_func(data_type, img_size, img_depth, forecasting_horizon,
        seq_len, model_type, batch_size, 
        max_epoch, optimizer, opt_learningrate,
        batchnorm_on, dropout_on, loss_func, earlystop_on,
        n_conv_layers, conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth,
        conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
        n_lstm_layers,  lstm_units, act_lstm, lstm_fc_units, n_fc_layers,
        last_fc_units, temp_type, traj_opt, normalization_opt):
    
    packed_args = {}
    # TODO: pack args as dict and call train_models with this dict
    train_model('test_datasize', 1, 0, packed_args) 

def main():    
    wait_job_request(sample_obj_func, True)

if __name__ == '__main__':
    main()