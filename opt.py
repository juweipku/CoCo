import argparse

parser = argparse.ArgumentParser()

# config
parser.add_argument('--device', type=str, default="gpu", help='device.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
parser.add_argument('--log', type=str, default="log.csv", help='log file name')

# pre-process
parser.add_argument('--n_input', type=int, default=500, help='input feature dimension')
parser.add_argument('--t', type=int, default=2, help="filtering time of Laplacian filters")

# network
parser.add_argument('--beta', type=float, default=1, help='focusing factor beta')
parser.add_argument('--dims', type=int, default=[1500], help='hidden unit')
parser.add_argument('--activate', type=str, default='ident', help='activate function')

# training
parser.add_argument('--runs', type=int, default=10, help='runs')
parser.add_argument('--epochs', type=int, default=400, help='training epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--acc', type=float, default=0, help='acc')
parser.add_argument('--nmi', type=float, default=0, help='nmi')
parser.add_argument('--ari', type=float, default=0, help='ari')
parser.add_argument('--f1', type=float, default=0, help='f1')

parser.add_argument('--k', type=int, default=64, help='the number of subspace')
parser.add_argument('--stage_num', type=int, default=10, help='the number of iterative optimization of EM algorithm')

args = parser.parse_args()
