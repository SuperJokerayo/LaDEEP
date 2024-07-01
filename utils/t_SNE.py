from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cal_tSNE(feature, target, n_components = 2):
    tsne = TSNE(n_components = n_components)

    feature_tsne = tsne.fit_transform(feature)

    plt.figure(figsize = (10, 10))
    plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c = target.astype(int), cmap = 'jet')
    plt.colorbar()
    plt.show()


