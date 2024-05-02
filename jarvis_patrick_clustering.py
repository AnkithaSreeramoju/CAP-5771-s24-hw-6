"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################
def display_clustering_plot(dataset, cluster_labels, plot_title):
    """
    Plot clustering results with points colored by cluster labels.
    
    Parameters:
    - dataset: A 2D numpy array of data points.
    - cluster_labels: Array of cluster labels corresponding to each data point.
    - plot_title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.title(plot_title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def calculate_sum_squared_errors(dataset, cluster_labels):
    """
    Calculate the sum of squared errors (SSE) for clustering.
    
    Parameters:
    - dataset: A 2D numpy array of data points.
    - cluster_labels: Array of cluster labels corresponding to each data point.
    
    Returns:
    - float: Total SSE for the clustering.
    """
    total_sse = 0.0
    for cluster_id in np.unique(cluster_labels):
        points = dataset[cluster_labels == cluster_id]
        center = np.mean(points, axis=0)
        total_sse += np.sum((points - center) ** 2)
    return total_sse

def calculate_adjusted_rand_index(actual_labels, predicted_labels):
    """
    Compute the adjusted Rand index to measure the similarity between two data clusterings.
    
    Parameters:
    - actual_labels: Array of actual labels.
    - predicted_labels: Array of predicted labels by the clustering algorithm.
    
    Returns:
    - float: Adjusted Rand index.
    """
    matrix = np.zeros((np.max(actual_labels) + 1, np.max(predicted_labels) + 1), dtype=np.int64)
    for i in range(len(actual_labels)):
        matrix[actual_labels[i], predicted_labels[i]] += 1

    sum_a = np.sum(matrix, axis=1)
    sum_b = np.sum(matrix, axis=0)
    total_samples = np.sum(matrix)
    sum_a_pairs = np.sum(sum_a * (sum_a - 1)) / 2
    sum_b_pairs = np.sum(sum_b * (sum_b - 1)) / 2

    sum_ab_pairs = np.sum(matrix * (matrix - 1)) / 2

    expected_index = sum_a_pairs * sum_b_pairs / total_samples / (total_samples - 1) + sum_ab_pairs ** 2 / total_samples / (total_samples - 1)
    max_index_possible = (sum_a_pairs + sum_b_pairs) / 2
    return (sum_ab_pairs - expected_index) / (max_index_possible - expected_index)

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    k = params_dict['k']  # Number of neighbors to consider
    s_min = params_dict['s_min']   # Similarity threshold

    num_samples = len(data)
    computed_labels = np.zeros(n, dtype=np.int32)

    for i in range(num_samples):
        # Calculate Euclidean distances from the current point to all others
        distances = cdist([data[i]], data, metric='euclidean')[0]

        # Get indices of the k nearest neighbors, excluding the point itself
        neighbor_indices = np.argsort(distances)[1:k+1]

        # Tally up the labels of the nearest neighbors
        neighbor_label_counts = np.bincount(true_labels[neighbor_indices])

        # Determine the label with the most occurrences among neighbors
        most_common_label = np.argmax(neighbor_label_counts)

        # Calculate similarity as the proportion of neighbors sharing the most common label
        neighbor_similarity = neighbor_label_counts[most_common_label] / k

        # Assign the label if the similarity meets or exceeds the threshold
        if neighbor_similarity >= s_min:
            computed_labels[i] = most_common_label + 1  # Increment label by 1 to avoid using 0 as a label

    # Calculate the Adjusted Rand Index for the clustering
    ARI = calculate_adjusted_rand_index(true_labels, predicted_labels)

    # Compute the Sum of Squared Errors for the clustering
    SSE = calculate_sum_squared_errors(data, predicted_labels)

    return computed_labels, SSE, ARI

def evaluate_hyperparameters(dataset, true_labels, neighbor_range, similarity_thresholds, trials):
    """
    Conduct a study to find the best hyperparameters for clustering based on Adjusted Rand Index.

    Parameters:
    - dataset: A numpy array of the input data points.
    - true_labels: True labels for the data points for evaluation.
    - neighbor_range: Range of 'k' values (number of neighbors) to test.
    - similarity_thresholds: Range of 's_min' values (similarity thresholds) to test.
    - trials: Number of trials to perform for each hyperparameter combination.

    Returns:
    - Tuple containing the best 'k' value, the best similarity threshold, and the highest average ARI observed.
    """
    highest_ari = -1
    best_num_neighbors = None
    best_similarity_threshold = None

    for num_neighbors in neighbor_range:
        for similarity_threshold in similarity_thresholds:
            cumulative_ari = 0
            for _ in range(trials):
                params = {'k': num_neighbors, 's_min': similarity_threshold}
                _, ari_score, _ = jarvis_patrick(dataset, true_labels, params)
                cumulative_ari += ari_score

            average_ari = cumulative_ari / trials
            if average_ari > highest_ari:
                highest_ari = average_ari
                best_num_neighbors = num_neighbors
                best_similarity_threshold = similarity_threshold

    return best_num_neighbors, best_similarity_threshold



def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').


    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')

    # Randomly sample data points
    random_indices = np.random.choice(len(cluster_data), size=5000, replace=False)
    subset_data = cluster_data[random_indices]
    subset_labels = cluster_labels[random_indices]

    subset_data = subset_data[:1000]
    subset_labels = subset_labels[:1000]

    # Define hyperparameters to test
    neighbor_values = [3, 4, 5, 6, 7, 8]
    similarity_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iterations = 10

    optimal_k, optimal_s_min = evaluate_hyperparameters(subset_data, subset_labels, neighbor_values, similarity_values, iterations)

    
    params_dict = {'k': optimal_k, 's_min': optimal_s_min}
    
    # Store clustering results
    groups = {}
    plot_values = {}

    # Apply best hyperparameters on data slices
    for i in range(5):
        slice_data = cluster_data[i * 1000: (i + 1) * 1000]
        slice_labels = cluster_labels[i * 1000: (i + 1) * 1000]
        labels, sse, ari = jarvis_patrick(slice_data, slice_labels, {'k': optimal_k, 's_min': optimal_s_min})
        groups[i] = {"s_min": optimal_s_min, "k": optimal_k, "ARI": ari, "SSE": sse}
        plot_values[i] = {"labels": labels, "ARI": ari, "SSE": sse}

    # Identify the dataset with the highest ARI and lowest SSE
    best_ari_index = max(plot_values, key=lambda idx: plot_values[idx]['ARI'])
    lowest_sse_index = min(plot_values, key=lambda idx: plot_values[idx]['SSE'])

    pdf_output = pdf.PdfPages("jarvis_patrick_clustering_results.pdf")

    # Plot clustering results
    def plot_clustering_results(data, labels, title, subtitle):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.title(f'Clustering for Dataset {best_dataset_index_sse} (Lowest SSE) with k value :{best_k} and sigma: 0.1')
        plt.suptitle('Spectral Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster')
        plt.grid(True)
        pdf_output.savefig()
        plt.close()

    plot_clustering_results(cluster_data[best_ari_index * 1000: (best_ari_index + 1) * 1000],
                            plot_values[best_ari_index]['labels'],
                            f'Best Clustering Result (Highest ARI) with k={optimal_k} and s_min={optimal_s_min}',
                            'Jarvis - Patrick Clustering')

    plot_clustering_results(cluster_data[lowest_sse_index * 1000: (lowest_sse_index + 1) * 1000],
                            plot_values[lowest_sse_index]['labels'],
                            f'Best Clustering Result (Lowest SSE) with k={optimal_k} and s_min={optimal_s_min}',
                            'Jarvis - Patrick Clustering')

    pdf_output.close()

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    # plot_ARI = plt.scatter([1,2,3], [4,5,6])
    # plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_dev_ari = np.std(ari_values) 
    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_dev_ari

    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_dev_sse = np.std(sse_values)

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_dev_sse


    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
