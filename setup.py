from opt import args


def setup_args():
    args.device = "cuda"
    args.acc = args.nmi = args.ari = args.f1 = 0
    args.consistency_t = 0.02
    args.alpha_value = 0.2

    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'leakyrelu'
        args.beta = 10

    elif args.dataset == 'cite':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'leakyrelu'
        args.beta = 9

    elif args.dataset == 'amap':
        args.t = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.beta = 6

    elif args.dataset == 'bat': # nor, 1
        args.t = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'sigmoid'
        args.beta = 5

    elif args.dataset == 'eat': # nor, 2/3
        args.t = 3
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.beta = 4

    elif args.dataset == 'uat': # t=4 nor 1 t=3youtidu
        args.t = 3
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.beta = 3

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.beta = 1

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("focusing factor: {}".format(args.beta))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args
