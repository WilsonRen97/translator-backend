from sklearn.cluster import DBSCAN
import numpy as np


def cluster_data(data, eps=150, min_samples=3, x_scale=0.75, y_scale=1.5):
    """
    x_scale > 1.0 stretches horizontal distances
    x_scale < 1.0 compresses horizontal distances
    """
    points = np.array([[obj['x'] * x_scale, obj['y'] * y_scale]
                      for obj in data])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    for obj, label in zip(data, labels):
        obj['cluster'] = int(label)
    return data
