from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


class BisectingKMeans:
    N_CLUSTERS_PER_ITER = 2

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                 precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated',
                 algorithm='auto'):
        self.kmeans = KMeans(2, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x,
                             n_jobs, algorithm)
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None
        self.current_n_clusters = 0
        self.X = None
        self.y = None
        self.sample_weight = None
        self.inertia_ = None

    def fit(self, X, y=None, sample_weight=None):
        """
        Compute bisecting k-means clustering
        :param X: dataframe of shape (n_samples, n_features. Training instances to cluster.
        :param y: Not used, present here for API consistency.
        :param sample_weight: The weights of each observation in X. If None, all observations have equal weight.
        :return: self
        """
        # set all data as a single cluster
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.labels_ = np.zeros((X.shape[0],), dtype=int)
        self.current_n_clusters = 1
        self.cluster_centers_ = np.array([np.mean(X, 0)]*self.n_clusters)

        while self.current_n_clusters < self.n_clusters:
            # find cluster to bisect
            if self.current_n_clusters == 1:
                bisect_label = 0
            else:
                # measure SSE & get max
                sses = self.wcss()
                bisect_label = np.argmax(sses)

            # bisect using kmeans
            label_x = self.X.iloc[np.where(self.labels_ == bisect_label)]
            label_y = None
            label_sample_weight = None
            if self.y:
                label_y = self.y[np.where(self.labels_ == bisect_label)]
            if self.sample_weight:
                label_sample_weight = self.sample_weight[np.where(self.labels_ == bisect_label)]
            self.kmeans.fit(label_x, label_y, label_sample_weight)
            self.labels_[self.labels_ == bisect_label] = np.where(self.kmeans.labels_ == 0,
                                                                  bisect_label, self.current_n_clusters)
            self.cluster_centers_[bisect_label] = self.kmeans.cluster_centers_[0]
            self.cluster_centers_[self.current_n_clusters] = self.kmeans.cluster_centers_[1]
            self.current_n_clusters += 1

        # calculate inertia after fitting data
        self.inertia_ = np.sum(self.wcss())

        return self

    def predict(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to.
        :param X: dataframe of shape (n_samples, n_features). New data to predict.
        :param sample_weight: Ignored. Present for API consistency.
        :return: ndarray of shape (n_samples,). Index of the cluster each sample belongs to.
        """
        X = np.array(X)
        predicted_y = np.array([0]*X.shape[0])

        for i in range(X.shape[0]):
            x = X[i]
            distances = np.linalg.norm(self.cluster_centers_ - x, ord=2, axis=1)
            predicted_y[i] = np.argmin(distances)

        return predicted_y

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: dataframe of shape (n_samples, n_feature). New data to transform.
        :param y: Not used, present here for API consistency.
        :param sample_weight: The weights of each observation in X. If None, all observations have equal weight.
        :return: Index of the cluster each sample belongs to.
        """
        self.fit(X, y, sample_weight)
        return self.labels_

    def wcss(self):
        """
        Compute the within-cluster sum of squared errors.
        :return: ndarray of shape (current_n_clusters,). Sum of squared errors within each cluster.
        """
        assert self.labels_ is not None
        sse_arr = np.zeros((self.current_n_clusters,))

        for l in range(self.current_n_clusters):
            label_x = self.X.iloc[np.where(self.labels_ == l)]
            centroid = np.mean(label_x, 0)
            sse = np.sum(np.sum(np.square(label_x - centroid)))
            sse_arr[l] = sse

        return sse_arr

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        :param params: dict containing the estimator parameters.
        :return: self
        """
        self.__dict__.update(params)
