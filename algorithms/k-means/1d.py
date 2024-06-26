import logging
from itertools import islice
from typing import Generator, Never

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    format="[%(funcName)s - %(lineno)d]: %(message)s", level=logging.INFO
)

INT_TYPE = np.int16
FLOAT_TYPE = np.float32
NP_FLOAT_ARRAY = NDArray[FLOAT_TYPE]
NP_INT_ARRAY = NDArray[FLOAT_TYPE]


def generate_input_data(n: int, rand_gen: Generator[float, None, None]) -> NP_INT_ARRAY:
    return np.array([r for r in islice(rand_gen, n)], dtype=INT_TYPE)


def random_generator(low: int, high: int) -> Generator[float, None, Never]:
    while 1:
        yield np.random.randint(low, high)
    assert False


def initialise_cluster_centroids(
    n_clusters: int, rand_gen: Generator[float, None, Never]
) -> NP_FLOAT_ARRAY:
    while (
        len(
            cluster_centroids := np.array(
                [*set(r for r in islice(rand_gen, n_clusters))], dtype=FLOAT_TYPE
            )
        )
        != n_clusters
    ):
        pass
    return cluster_centroids


def assign_cluster_indices(
    input_data: NP_INT_ARRAY, cluster_centroids: NP_FLOAT_ARRAY
) -> NP_INT_ARRAY:
    return np.array(
        [
            (distances := [np.linalg.norm((datum - c)) for c in cluster_centroids]).index(
                min(distances)
            )
            for datum in input_data
        ],
        dtype=INT_TYPE,
    )


def compute_cluster_sizes(
    cluster_indices: NP_INT_ARRAY, n_clusters: int
) -> NP_INT_ARRAY:
    return np.array(
        [
            len([idx for idx in cluster_indices if idx == cluster_index])
            for cluster_index in range(n_clusters)
        ]
    )


def reinitialise_empty_clusters(
    input_data: NP_INT_ARRAY,
    cluster_indices: NP_INT_ARRAY,
    cluster_centroids: NP_FLOAT_ARRAY,
    rand_gen: Generator[float, None, Never],
) -> tuple[NP_INT_ARRAY, NP_FLOAT_ARRAY]:
    cluster_sizes = compute_cluster_sizes(
        cluster_indices, len(cluster_centroids))

    while any([size == 0 for size in cluster_sizes]):
        for cluster_index, cluster_size in enumerate(cluster_sizes, 0):
            if cluster_size == 0:
                cluster_centroids[cluster_index] = next(rand_gen)
                cluster_indices = assign_cluster_indices(
                    input_data, cluster_centroids)
                cluster_sizes = compute_cluster_sizes(
                    cluster_indices, len(cluster_centroids)
                )
                break
    return cluster_indices, cluster_centroids


def update_cluster_centroids(
    input_data: NP_INT_ARRAY, cluster_indices: NP_INT_ARRAY, n_clusters
) -> NP_FLOAT_ARRAY:
    return np.array(
        [
            np.array(
                [
                    datum
                    for index, datum in zip(cluster_indices, input_data)
                    if index == cluster_index
                ]
            ).mean()
            for cluster_index in range(n_clusters)
        ]
    )


def main(
    input_data: NP_INT_ARRAY,
    n_clusters: int,
    epsilon: float,
    rand_gen: Generator[float, None, Never],
):
    logging.info(f"{input_data=}")

    cluster_centroids = initialise_cluster_centroids(n_clusters, rand_gen)

    logging.info(f"Initialised {n_clusters} clusters")

    cluster_indices = assign_cluster_indices(input_data, cluster_centroids)

    while True:
        old_cluster_centroids = cluster_centroids

        # Update clusters with fixed cluster_centroids
        cluster_indices = assign_cluster_indices(input_data, cluster_centroids)
        cluster_indices, cluster_centroids = reinitialise_empty_clusters(
            input_data, cluster_indices, cluster_centroids, rand_gen
        )

        # Update clusters with fixed cluster_indices
        cluster_centroids = update_cluster_centroids(
            input_data, cluster_indices, n_clusters
        )

        if (np.linalg.norm(old_cluster_centroids - cluster_centroids) < epsilon).all():
            break

    logging.info(f"Found clusters {cluster_centroids=}")


if __name__ == "__main__":
    LOW, HIGH = -10, 11
    N_CLUSTERS = 3
    EPSILON = 0.0001

    rand_gen = random_generator(LOW, HIGH)
    input_data = generate_input_data(20, rand_gen)
    main(input_data, N_CLUSTERS, EPSILON, rand_gen)
