import json
from datetime import datetime
import torch.nn as nn
import torch
import os

from utils import get_data, create_data_loaders, plot_losses, SlidingWindowDataset

from models.model import Aformer
from lib.prediction import Predictor
from lib.training import Trainer
import argparse
from utils import str2bool 
import seaborn as sns
import sys

import numpy as np
import random


# python test_pre.py --dataset SWaT --d_model 8 --d_ff 8
# python test_pre.py --dataset SWaT --d_model 8 --d_ff 16 --use_gcn True


class Dict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value



if __name__ == "__main__":
    # id = '14112021_164509'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="output/SWaT_10/15112021_204207")
    parser.add_argument("--only_graph", type=str2bool, default=False)

    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default='model')
    parser.add_argument("--topk", type=int, default=2)
    # # -- Data params ---
    
    test_args = parser.parse_args()
    load_path = test_args.id
    config_path = f"{load_path}/config.txt"
    with open(config_path, 'r') as f:
        args = json.load(f)
        args = Dict(args)

    # 随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    args_summary = str(args.__dict__)
    # print(args_summary)
    print('use graph', args.use_graph)
# data load
    group_index = args.group[0]
    index = args.group[2:]
    if args.dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=args.normalize)
    elif args.dataset in ['MSL', 'SMAP', 'SWaT', 'WADI', 'SWaT_10', 'WADI_10']:
        output_path = f'output/{args.dataset}'
        (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=args.normalize)
    else:
        raise Exception(f'Dataset "{args.dataset}" not available.')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    y_test = y_test
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    enc_in = x_train.shape[1]
    get_target_dims = {"SMAP":[0], "MSL":[0], "SMD":None, 'SWaT':None, 'WADI':None,  'SWaT_10':None, 'WADI_10':None}
    target_dims = get_target_dims[args.dataset]

    if target_dims is None:
        c_out = enc_in
        print(f"Will forecast and reconstruct all {enc_in} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        c_out = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        c_out = len(target_dims)
    train_dataset = SlidingWindowDataset(x_train, args.seq_len, enc_in)
    test_dataset = SlidingWindowDataset(x_test, args.seq_len, enc_in)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, args.batch_size, args.val_split, args.shuffle_dataset, test_dataset=test_dataset
    )

    model = Aformer(1, c_out,
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, e_layers=args.e_layers, start_len=args.start_len, 
            embed='fixed', freq='h', dropout=args.dropout, attn=args.attn, factor=args.factor, k=args.k, 
            activation=args.activation, output_attention = False, use_gcn = args.use_graph, num_nodes=enc_in, AE = args.use_AE, seq_len=args.seq_len)

    


    model.load_state_dict(torch.load(f"{load_path}/model.pt", map_location='cuda'))
    if args.use_graph:
        adp = model.gc(torch.arange(model.num_nodes))
        sns.set(font_scale=1.5)
        sns.set_context({"figure.figsize":(8,8)})
        fig = sns.heatmap(data=adp.detach().numpy(),square=True) 
        scatter_fig = fig.get_figure()
        scatter_fig.savefig(f"{load_path}/apt", dpi = 400)
    if test_args.only_graph:
        sys.exit(0)

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "WADI":  (0.9950, 0.001),
        "SWaT":  (0.9950, 0.001),
        "WADI_10":  (0.9950, 0.001),
        "SWaT_10":  (0.9950, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "WADI":1, "SWaT":1, "WADI_10":1, "SWaT_10":1}
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    reg_level = reg_level_dict[key]

    # trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': args.dataset,
        "target_dims": target_dims,
        'scale_scores': test_args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': test_args.dynamic_pot,
        "use_mov_av": test_args.use_mov_av,
        "gamma": test_args.gamma,
        "reg_level": reg_level,
        "save_path": load_path,
    }
    best_model = model.to('cuda')

    # Creating a new summary-file each time when new prediction are made with a pre-trained model
    count = 0
    for filename in os.listdir(load_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"
    predictor = Predictor(
        best_model,
        args.seq_len,
        enc_in,
        prediction_args,
        batch_size = args.batch_size,
        run_mode=args.run_mode,
        summary_file_name=summary_file_name,
        topk = test_args.topk
    )

    label = y_test[args.seq_len:] if y_test is not None else None

    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    # args_path = f"{load_path}/config.txt"
    # with open(args_path, "w") as f:
    #     json.dump(args.__dict__, f, indent=2)
