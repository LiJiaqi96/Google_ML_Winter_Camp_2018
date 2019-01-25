from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering as Hierarchical
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

samples = np.load("/data8t/ljq/whale_data/whale_data/convert/dataset/feature_map.npy")
print(samples.shape)

# t-SNE
tsne_out = TSNE(perplexity=50).fit_transform(samples)

# t-SNE Plot
print(tsne_out.shape)
plt.scatter(tsne_out[:,0],tsne_out[:,1],c='deepskyblue',s=0.05)
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.title('tSNE Plot of All Images')
plt.show()

# K-means on t-SNE results
clus = DBSCAN(eps=0.5, min_samples=5, metric='euclidean').fit(tsne_out)
clu_labels = clus.labels_
idx_0 = np.where(clu_labels == 0)
idx_1 = np.where(clu_labels == 1)
idx_2 = np.where(clu_labels == 2)
idx_3 = np.where(clu_labels == 3)
idx_4 = np.where(clu_labels == 4)
idx_5 = np.where(clu_labels == 5)
plt.figure()
plt.scatter(tsne_out[idx_0,0],tsne_out[idx_0,1],c='yellow',s=0.05)
plt.scatter(tsne_out[idx_1,0],tsne_out[idx_1,1],c='green',s=0.05)
plt.scatter(tsne_out[idx_2,0],tsne_out[idx_2,1],c='red',s=0.05)
plt.scatter(tsne_out[idx_3,0],tsne_out[idx_3,1],c='blue',s=0.05)
plt.scatter(tsne_out[idx_4,0],tsne_out[idx_4,1],c='pink',s=0.05)
plt.scatter(tsne_out[idx_5,0],tsne_out[idx_5,1],c='brown',s=0.05)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot with E5 Subtypes')
plt.show()
