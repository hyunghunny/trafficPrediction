from wot.interface import *
from main import train_model
import wot.utils.hp_cfg as hp_cfg

@eval_task
def traffic_model_func(data_type, img_size, img_depth, forecasting_horizon,
        seq_len, model_type, batch_size, 
        max_epoch, optimizer, opt_learningrate,
        batchnorm_on, dropout_on, loss_func, earlystop_on,
        n_conv_layers, conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth,
        conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
        n_lstm_layers,  lstm_units, act_lstm, lstm_fc_units, n_fc_layers,
        last_fc_units, temp_type, traj_opt, normalization_opt):
        
    hyperparams = pack_args_to_dict(data_type, img_size, img_depth, forecasting_horizon,
        seq_len, model_type, batch_size, 
        max_epoch, optimizer, opt_learningrate,
        batchnorm_on, dropout_on, loss_func, earlystop_on,
        n_conv_layers, conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth,
        conv_fc_units, act_conv, pooling_on, pooling_size, conv_filter_size,
        n_lstm_layers,  lstm_units, act_lstm, lstm_fc_units, n_fc_layers,
        last_fc_units, temp_type, traj_opt, normalization_opt)
    
    train_model('test_datasize', 1, 0, hyperparams) 

def pack_args_to_dict(*args, cfg_file="hp_conf/traffic.json"):
    cfg = hp_cfg.read_config(cfg_file)
    if cfg != None:
        args_dict = {}
        if len(cfg.param_order) == len(args):
            num_params = len(cfg.param_order)
            for i in range(num_params):
                args_dict[cfg.param_order[i]] = args[i]
            #print(args_dict)
            return args_dict
        else:
            raise ValueError("Size of arguments {} is not equal to required parameters {}".format(len(args), len(cfg.param_order)))
    else:
        raise ValueError("Invalid range or value: {}".format(args))
def main():    
    wait_job_request(traffic_model_func, True)

if __name__ == '__main__':
    main()
