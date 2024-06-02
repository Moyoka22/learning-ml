import logging

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    format="[%(funcName)s - %(lineno)d]: %(message)s", level=logging.INFO)

INT_TYPE = np.int16
FLOAT_TYPE = np.float32
NP_FLOAT_ARRAY = NDArray[FLOAT_TYPE]
NP_INT_ARRAY = NDArray[FLOAT_TYPE]


def generate_input_data(n: int, low: int, high: int) -> NP_INT_ARRAY:
    return np.array([np.random.randint(low, high) for _ in range(n)], dtype=INT_TYPE)


def initialise_cluster_centroids(n_clusters: int, low: int, high: int) -> NP_FLOAT_ARRAY:
    while len(cluster_centroids := np.array([*set(np.random.randint(low, high) for _ in range(n_clusters))], dtype=FLOAT_TYPE)) != n_clusters:
        pass
    return cluster_centroids


def assign_cluster_indices(input_data: NP_INT_ARRAY, cluster_centroids: NP_FLOAT_ARRAY) -> NP_INT_ARRAY:
    return np.array([
        (distances := [np.sqrt((datum - c)**2)
         for c in cluster_centroids]).index(min(distances))
        for datum in input_data
    ], dtype=INT_TYPE)


def compute_cluster_sizes(cluster_indices: NP_INT_ARRAY, n_clusters: int) -> NP_INT_ARRAY:
    return np.array([len([idx for idx in cluster_indices if idx == cluster_index]) for cluster_index in range(n_clusters)])


def reinitialise_empty_clusters(input_data: NP_INT_ARRAY,
                                cluster_indices: NP_INT_ARRAY,
                                cluster_centroids: NP_FLOAT_ARRAY,
                                low: int,
                                high: int
                                ) -> tuple[NP_INT_ARRAY, NP_FLOAT_ARRAY]:
    cluster_sizes = compute_cluster_sizes(
        cluster_indices, len(cluster_centroids))

    while any([size == 0 for size in cluster_sizes]):
        for cluster_index, cluster_size in enumerate(cluster_sizes, 0):
            if cluster_size == 0:
                cluster_centroids[cluster_index] = np.random.randint(low, high)
                cluster_indices = assign_cluster_indices(
                    input_data, cluster_centroids)
                cluster_sizes = compute_cluster_sizes(
                    cluster_indices, len(cluster_centroids))
                break
    return cluster_indices, cluster_centroids


def update_cluster_centroids(input_data: NP_INT_ARRAY, cluster_indices: NP_INT_ARRAY, n_clusters) -> NP_FLOAT_ARRAY:
    return np.array([
        np.array([datum for index, datum in zip(cluster_indices, input_data)
                  if index == cluster_index]).mean()
        for cluster_index in range(n_clusters)
    ])


def main(input_data: NP_INT_ARRAY, n_clusters: int, low: int, high: int):

    logging.info(f"{input_data=}")

    cluster_centroids = initialise_cluster_centroids(n_clusters, low, high)

    logging.info(f"Initialised {n_clusters} clusters")

    cluster_indices = assign_cluster_indices(input_data, cluster_centroids)

    while True:
        old_cluster_centroids = cluster_centroids

        # Update clusters with fixed cluster_centroids
        cluster_indices = assign_cluster_indices(input_data, cluster_centroids)
        cluster_indices, cluster_centroids = reinitialise_empty_clusters(
            input_data, cluster_indices, cluster_centroids, low, high)

        # Update clusters with fixed cluster_indices
        cluster_centroids = update_cluster_centroids(
            input_data, cluster_indices, n_clusters)

        if (old_cluster_centroids == cluster_centroids).all():
            break

    logging.info(f"Found clusters {cluster_centroids=}")


if __name__ == '__main__':
    low, high = -10, 11
    n_clusters = 3
    input_data = generate_input_data(20, low, high)
    main(input_data, n_clusters, low, high)
