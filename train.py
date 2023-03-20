import json
from datetime import datetime
import torch.nn as nn
import torch
import os

import argparse
import seaborn as sns
import numpy as np
import random

from utils import get_data, get_target_dims, create_data_loaders, SlidingWindowDataset, get_value_from_train, str2bool, plot_losses

from models.model import Aformer
from lib.prediction import Predictor
from lib.training import Trainer


if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', help='random seed', type = int, default=0)

    # -- Data params ---
    parser.add_argument("--dataset", type=str, default="SMD")
    parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    # parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    # log
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)


    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)
    parser.add_argument("--topk", type=int, default=2)

    # model
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Informer encoder')
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--use_graph', type=str2bool, default=False, help='use graph')
    parser.add_argument('--k', type=int, default=0, help='antoencoder_k')
    parser.add_argument('--use_AE', type=str, default=None, help='AE/VAE')
    parser.add_argument('--start_len', type=int, default=5, help='')
    parser.add_argument('--subgraph_size', type=int, default=10, help='')
    parser.add_argument('--e_dim', type=int, default=10, help='')

    # train
    # parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=7, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--run_mode', type=str, default=None, help='fore/recon')

    parser.add_argument('--des', type=str, default='',help='exp description')
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    args_summary = str(args.__dict__)
    print('graph:', args.use_graph)

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
    
    # logs
    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    id = id+args.des
    save_path = f"{output_path}/{id}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    x_train = torch.from_numpy(x_train).float() # (T,N)
    x_test = torch.from_numpy(x_test).float()
    enc_in = x_train.shape[1]
    target_dims = get_target_dims(args.dataset)

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
        activation=args.activation, output_attention = False, use_gcn = args.use_graph, num_nodes=enc_in, AE = args.use_AE, subgraph_size=args.subgraph_size, node_dim=args.e_dim, seq_len=args.seq_len)
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    #     else:
    #         nn.init.uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model, optimizer, args.seq_len, enc_in, target_dims,
        args.train_epochs, args.batch_size, args.learning_rate,
        forecast_criterion, recon_criterion,
        args.use_gpu, save_path, log_dir,
        args.print_every, args.log_tensorboard,
        args_summary, patience=args.patience, debug = args.debug, run_mode=args.run_mode
    )
    trainer.fit(train_loader, val_loader)

    # plot_losses(trainer.losses, save_path=save_path, plot=False)
    trainer.load(f"{save_path}/model.pt")

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    trainer.logger.info(f"Test forecast loss: {test_loss[0]:.5f}")
    trainer.logger.info(f"Test reconstruction loss: {test_loss[1]:.5f}")
    trainer.logger.info(f"Test total loss: {test_loss[2]:.5f}")

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
        "SWaT_10":  (0.9950, 0.001),
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
        'scale_scores': args.scale_scores,   # 是否在get_score中使用中值和iqr进行正规化
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,   # POT_eval中的一个参数
        "use_mov_av": args.use_mov_av,    # 是否使用滑动窗口平均
        "gamma": args.gamma,      # 重构分数占比
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model

    # if args.use_graph:
    #     if args.use_gpu:
    #         adp = model.gc(torch.arange(model.num_nodes).to('cuda')).cpu()
    #     else:
    #         adp = model.gc(torch.arange(model.num_nodes)).cpu()
    #     adj = adp + torch.eye(adp.size(0))
    #     d = adj.sum(1)
    #     dv = d
    #     adj = adj / dv.view(-1, 1)
    #     mask = torch.zeros(adj.size(0), adj.size(0))
    #     mask.fill_(float('0'))
    #     s1,t1 = (adj).topk(5, 1)
    #     mask.scatter_(1,t1,s1.fill_(1))
    #     np.savetxt(f"{save_path}/adj.txt", mask.detach().numpy(), fmt="%d")
    #     sns.set(font_scale=1.5)
    #     sns.set_context({"figure.figsize":(8,8)})
    #     fig = sns.heatmap(data=mask.detach().numpy(),square=True) 
    #     scatter_fig = fig.get_figure()
    #     scatter_fig.savefig(f"{save_path}/apt", dpi = 400)

    #     if args.use_gpu:
    #         embeddings = model.gc.emb1(torch.arange(model.num_nodes).to('cuda'))
    #     else:
    #         embeddings = model.gc.emb1(torch.arange(model.num_nodes))
    #     torch.save(embeddings, f"{save_path}/embeddings.pt")
    predictor = Predictor(
        best_model,
        args.seq_len,
        enc_in,
        prediction_args,
        batch_size = args.batch_size,
        run_mode=args.run_mode, 
        topk = args.topk
    )

    label = y_test[args.seq_len:] if y_test is not None else None
    x_value = get_value_from_train(x_train, val_split = args.val_split)
    # predictor.predict_anomalies(x_train, x_test, label, value=x_value)
    predictor.predict_anomalies(x_train, x_test, label)
