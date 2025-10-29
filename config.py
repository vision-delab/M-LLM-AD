import os

share_config = {'data_root': './data'}

class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'configutation' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None):
    share_config['dataset_name'] = args.dataset
    share_config['class_name'] = args.class_name
    share_config['set_image'] = args.set_image
    share_config['anomaly_detect'] = args.anomaly_detect
    share_config['visualization'] = args.visualization
    share_config['prompt_ver'] = args.prompt_ver
    share_config['llm_model'] = args.llm_model
    share_config['top_k'] = args.top_k
    share_config['group_length'] = args.group_length
    share_config['interval'] = args.interval
    share_config['grouping'] = args.grouping


    if share_config['dataset_name'] == 'synthetic':
        share_config['data_path'] = share_config['data_root']+f'/synthetic/{args.class_name}/eval/data.pkl'

    return dict2class(share_config)