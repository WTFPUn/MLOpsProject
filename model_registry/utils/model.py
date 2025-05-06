from sklearn.preprocessing import StandardScaler
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from sklearn.cluster import OPTICS
import umap.umap_ as umap # may have the version problem update as need: https://github.com/lmcinnes/umap/issues/828

class Embedding_model:
    def __init__(self, model_name= 'BAAI/bge-m3'):
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True
        )

    def embed(self, sentences: list):
        """this code will acceot list of sentences and then embed the vector and transform it before clustering

        Args:
            sentences (list): list of raw text (news)
        """
        # extract sentences
        outputs = self.model.encode(sentences, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        colbert_vecs = outputs['colbert_vecs']
        # Pooled vectors from colbert_vecs
        pooled_vecs = np.array([vec.mean(axis=0) for vec in colbert_vecs])

        # ✅ Standardize
        pooled_vecs = StandardScaler().fit_transform(pooled_vecs)
        return pooled_vecs

    def predict(self, embedding):
        # ✅ UMAP → 2D
        umap_2d = umap.UMAP(n_neighbors=2, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embedding)

        # ✅ OPTICS Clustering
        optics = OPTICS(min_samples=2, metric='euclidean', xi=0.05, min_cluster_size=2)
        labels = optics.fit_predict(umap_2d)
        return labels
    
