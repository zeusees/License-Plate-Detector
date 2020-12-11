# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[24, 48], [96, 192], [384, 768]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 48,
    'ngpu': 1,
    'epoch': 50,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

