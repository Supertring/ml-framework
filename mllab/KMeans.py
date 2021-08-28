import numpy as np
from .metrics import euclidean_distances


class KMeans:
    """
     KMeans.

     Parameters
     -----------
     _x             : array
                        input training points, size N:D
     _k             : int
                        number of centroids
     _cluster_center: list
                        computed cluster centers
     _max_iter      : int
                        maximum number of iteration of kmeans to reduce loss , default value : 300
     _tol           : float
                        threshold value for convergence of kmeans, default : 0.0001
     _clusters      : dictionary
                        eg: {cluster_number : list of points in cluster}
     _closest_cluster_ids : list
                        list of cluster ids, that will exactly be cluster value of input training set.
     _distance_method:
     """

    def __init__(self):
        self._x = []
        self._k = 1
        self._cluster_centers = []
        self._max_iter = 300
        self._tol = 1e-4
        self._clusters = dict()
        self._closest_cluster_ids = []
        self._distance_method = ""

    """
    FUNCTION NAME: random_initial_centroids
    Args:   x (array) : input data points
            k (int)   : number of centroids  
            
    Task :  Choose k unique random points from x datasets, 
            these points will be used as initial centroids
    
    Returns :    (numpy.ndarray), array of k unique centroid points    
    """
    def random_initial_centroids(self, x, k):
        # number of total training datasets
        n_rows, n_columns = x.shape
        # randomly get k unique sample points
        unique_centroids = np.random.choice(range(0, n_rows), k, replace=False)
        # assign randomly chosen points to centroids
        centroids = x.iloc[unique_centroids]
        # return centroids
        return np.array(centroids)

    """
    FUNCTION NAME: compute_clusters
    Args :   x (array)               : input data points
             centroids (np.array)    : k no. of points computed from x dataset
             distance_method         : name of method to calculate distance between centroids and data points, 
                                       default: euclidean_distances
                                       function takes two arguments, x: input datasets, centroids : k centroid points
                                       return: distance between x  and centroids.
    Task :  Computes new k centroids and assign each data points from x to new centroids
    
    Returns :   cluster : dict{cluster_number : list of points in cluster}
                closest_cluster_ids : list of cluster ids, that will exactly be cluster value of input training set.
    """

    def compute_clusters(self, x, centroids, distance_method):
        x = np.array(x)
        k = centroids.shape[0]
        # cluster dictionary
        clusters = {}
        distance_matrix = distance_method(x, centroids)
        # assign points to the closest cluster
        closest_cluster_ids = np.argmin(distance_matrix, axis=1)
        for i in range(k):
            clusters[i] = []
        # assign points for cluster
        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(x[i])
        return clusters, closest_cluster_ids

    """
    FUNCTION NAME : train
    Args :  x               (array): input training points, size N:D
            k               (int)  : number of centroids
            max_iter        (int)  : maximum number of iteration of kmeans to reduce loss , default value : 300
            tol             (float): threshold value for convergence of kmeans, default : 0.0001
            distance_method        : name of method to calculate distance between centroids and data points, 
        
    Task :  performs k-means algorithm on datasets x
    """
    def train(self, x, k, max_iter=300, tol=1e-4, distance_method=euclidean_distances):
        self._x = x
        self._k = k
        self._max_iter = max_iter
        self._tol = tol
        self._cluster_centers = self.random_initial_centroids(self._x, k)
        self._distance_method = distance_method

        for i in range(self._max_iter):
            previous_cluster_center = self._cluster_centers
            self._clusters, self._closest_cluster_ids = self.compute_clusters(self._x, previous_cluster_center, self._distance_method)
            self._cluster_centers = np.array(
                [np.mean(self._clusters[key], axis=0, dtype=np.array(self._x).dtype) for key in
                 sorted(self._clusters.keys())])

    """
    FUNCTION NAME: infer
    Returns :   _clusters      : dictionary
                                    eg: {cluster_number : list of points in cluster}
                _closest_cluster_ids : list
                                    list of cluster ids, that will exactly be cluster value of input training set.
    """
    def infer(self):
        return self._clusters, self._closest_cluster_ids
