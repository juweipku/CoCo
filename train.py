from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import compact_net
from consistency_loss import Consist_net


if __name__ == '__main__':

    # setup hyper-parameter
    args = setup_args()
    file_name = args.log

    # record results
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    print("ACC,   NMI,   ARI,   F1", file=file)
    file.close()
    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    # ten runs with different random seeds
    for args.seed in range(args.runs):
        # record results

        # fix the random seed
        setup_seed(args.seed)

        # load graph data
        X, y, A, node_num, cluster_num = load_graph_data(args.dataset, show_details=False)
        Ad = diffusion_adj(A, mode="ppr", transport_rate=args.alpha_value)
        A, Ad = torch.tensor(A).float(), torch.tensor(Ad).float()

        # apply the laplacian filtering
        X_l = laplacian_filtering(A, X, args.t) # Local
        X_g = laplacian_filtering(Ad, X, args.t) # Global

        # pre-test
        args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X_l, y, cluster_num)
        
        # args.consistency_memory_size = 128000
        args.consistency_memory_size = X.shape[0] * 10
        consistency = Consist_net(args)
        # build our network
        CoCo = compact_net(input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, 
                        k=args.k, stage_num=args.stage_num, beta=args.beta, cluster_num=cluster_num)

        optimizer = optim.Adam(CoCo.parameters(), lr=args.lr)

        # load data to device
        A, CoCo, X_l, X_g = map(lambda x: x.to(args.device), (A, CoCo, X_l, X_g))

        best_Z = None
        # training
        for epoch in tqdm(range(1000), desc="training..."):
            CoCo.train()

            Z1, Z2 = CoCo(X_l, X_g, A)
            loss = 0.5 * (consistency(Z1, Z2) + consistency(Z2, Z1))
            
            loss.backward()
            optimizer.step()

            # testing
            if epoch % 1 == 0:
                # evaluation mode
                CoCo.eval()

                # encoding
                Z1, Z2 = CoCo(X_l, X_g, A)

                # fusion and testing
                Z = (Z1 + Z2) / 2
                acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                # recording
                if acc >= args.acc:
                    args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                    best_Z = Z
                    best_P = P

        print("Training complete")

        # record results
        file = open(file_name, "a+")
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
        file.close()
        acc_list.append(args.acc)
        nmi_list.append(args.nmi)
        ari_list.append(args.ari)
        f1_list.append(args.f1)

        # np.savez('./TSNE/tsne_embedding/z_predicty_y'+'_'+args.dataset+'_'+str(args.seed)+'.npz', best_Z.cpu().detach().numpy(), best_P, y)

    # record results
    acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
    file = open(file_name, "a+")
    print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
    file.close()
