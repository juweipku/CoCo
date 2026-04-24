import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import exists
from sklearn.manifold import TSNE

# from utils import eva
# import opt


def pdf_format():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['axes.linewidth'] = 1
    matplotlib.rcParams['axes.edgecolor'] = 'Grey'

    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18

    params = {'font.family': 'Times New Roman',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal'  # or 'blod'
              }
    matplotlib.rcParams.update(params)

def plot_tsne(_K_classes, node_rep: np.ndarray, node_prelbl: np.ndarray, seedn, sample_num, seed):

    assert node_rep.shape[0] == node_prelbl.shape[0]

    # sampling
    np.random.seed(seedn)
    sample_index = np.random.randint(0, node_rep.shape[0], sample_num)
    sample_embeds = node_rep[sample_index]
    sample_labels = node_prelbl[sample_index]

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    # remove outlier
    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)
    idx = []
    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean > 3 * std).all():
            idx.append(i)
            # np.delete(ts_embeds, i)
        
    ts_embeds = np.delete(ts_embeds, idx).reshape(-1,2)
    sample_labels = np.delete(sample_labels, idx)

    # # normalization
    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    # Plot
    # colors = np.array(['coral', 'cornflowerblue', 'darkseagreen', 'pink', 'darkgrey', 'c', 'mediumpurple', 'peru'])
    # colors = np.array(['red','steelblue','limegreen','hotpink','yellow','darkorchid','darkorange','saddlebrown'])
    colors = np.array([(244/256,0,0),(15/256,107/256,175/256),(19/256,167/256,43/256),(146/256,48/256,151/256),
                (255/256,94/256,0/256),(255/256,255/256,0),(161/256,61/256,19/256),(255/256,92/256,181/256)])
    markers = np.array(['$0$','$1$','$2$','$3$','$4$','$5$','$6$','$7$'])
    fig, axis = plt.subplots()
    if name == 'amap' or name == 'dblp':
        laps = 10
    else:
        laps = 1  # lap > 1 is helpful when there's too much overlapping
    gap = norm_ts_embeds.shape[0] // laps
    for bg in range(laps):
        st = bg * gap
        axis.scatter(norm_ts_embeds[st: st + gap, 0], norm_ts_embeds[st: st + gap, 1], s=10, \
            c=colors[sample_labels[st: st + gap]], marker='.')
        # for i in range(gap):
        #     axis.scatter(norm_ts_embeds[st+i, 0], norm_ts_embeds[st+i, 1], s=20, \
        #             c=colors[sample_labels[st+i]], marker=markers[sample_labels[st+i]], linewidths=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig("./TSNE/tsne_output/amap/tsne_scatter_"+name+'_'+str(seed)+'_'+str(seedn)+".png", bbox_inches = 'tight')
    # plt.savefig("./tsne_scatter_"+name+'_'+str(seed)+'_'+str(seedn)+".pdf", bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    pdf_format()
    # if name == 'acm':
    #     sample_num = 2500
    #     seedn = 5
    # elif name == 'amap':
    #     sample_num = 7000
    #     seedn = 2
    # elif name == 'cite':
    #     sample_num = 3000
    #     seedn = 1
    # elif name == 'dblp':
    #     sample_num = 3500
    #     seedn = 1
    # elif name == 'hhar':
    #     sample_num = 10000
    #     seedn = 5
    # else:
    #     print("Please add the dataset!")

    name = 'amap'
    seed = 7
    sample_num = 2000
    # seedn = 0

    model_output = np.load('./TSNE/tsne_embedding/z_predicty_y'+'_'+name+'_'+str(seed)+'.npz')
    node_hid = model_output['arr_0'] # Shape: [n_nodes, hid_dim]
    node_prelabel = model_output['arr_1'] # Shape: [n_nodes, ]
    K_classes = np.unique(node_prelabel)

    for seedn in range(10):
        plot_tsne(K_classes, node_hid, node_prelabel, seedn, sample_num, seed)

