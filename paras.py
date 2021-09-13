import argparse

parser = argparse.ArgumentParser(description='Train DHGC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', default='dblp', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dims_en', default=[256, 16], type=list)
parser.add_argument('--dims_de', default=[16, 256], type=list)
parser.add_argument('--epochs', default=100, type=int)

parser.add_argument('--n_cluster', default=4, type=int)
parser.add_argument('--d_input', default=334, type=int)

parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--wd', default=0.0005, type=float)
parser.add_argument('--eta', default=0.0001, type=float)
parser.add_argument('--lambdas', default=3.0, type=float)
parser.add_argument('--gamma', default=1.0, type=float)

parser.add_argument('--log_save_path', default='log', type=str)
parser.add_argument('--model_path', default='pkl', type=str)
parser.add_argument('--graph_path', default='txt', type=str)
args = parser.parse_args()

if args.name == 'acm':
    args.n_cluster = 3
    args.d_input = 1870
    args.lr = 0.005
    args.wd = 0.005
    args.eta = 0.0001
    args.lambdas = 10.0
    args.gamma = 0.5

if args.name == 'cite':
    args.n_cluster = 6
    args.d_input = 3703
    args.lr = 0.005
    args.wd = 0.005
    args.eta = 0.0005
    args.lambdas = 5.0
    args.gamma = 0.7

if args.name == 'amazon':
    args.n_cluster = 8
    args.d_input = 745
    args.lr = 0.005
    args.wd = 0.0005
    args.eta = 0.00001
    args.lambdas = 5.0
    args.gamma = 1

args.graph_path = './data/{}_graph.txt'.format(args.name)
args.log_save_path = './result/train/log/{}.log'.format(args.name)
args.model_path = './result/pre/model/{}.pkl'.format(args.name)