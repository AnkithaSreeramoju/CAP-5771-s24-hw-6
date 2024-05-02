"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################
def calculate_SSE(dataset, cluster_labels):
    """
    Compute the Sum of Squared Errors (SSE) for the given clustering.

    Parameters:
    - dataset: The dataset as a numpy array.
    - cluster_labels: An array of cluster labels for each data point.

    Returns:
    - float: The computed SSE value.
    """
    sse_total = 0.0
    for label in np.unique(cluster_labels):
        points_in_cluster = dataset[cluster_labels == label]
        center_of_cluster = np.mean(points_in_cluster, axis=0)
        sse_total += np.sum((points_in_cluster - center_of_cluster) ** 2)
    return sse_total


def display_clustering_results(dataset, cluster_labels, plot_title):
    """
    Plot the clustering results.

    Parameters:
    - dataset: The dataset as a numpy array.
    - cluster_labels: An array of cluster labels for each data point.
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


def gaussian_similarity(x, y, sigma):
    """
    Compute the Gaussian similarity between two points.

    Parameters:
    - x, y: Points as numpy arrays.
    - sigma: Standard deviation of the Gaussian.

    Returns:
    - float: Gaussian similarity between x and y.
    """
    distance_squared = np.sum((x - y) ** 2)
    return np.exp(-distance_squared / (2 * sigma ** 2))

def compute_adjusted_rand_index(true_labels, predicted_labels):
    """
    Calculate the adjusted Rand index.

    Parameters:
    - true_labels: True labels of the data points.
    - predicted_labels: Predicted labels by the clustering algorithm.

    Returns:
    - float: Adjusted Rand index value.
    """
    matrix = np.zeros((np.max(true_labels) + 1, np.max(predicted_labels) + 1), dtype=np.int64)
    for i in range(len(true_labels)):
        matrix[true_labels[i], predicted_labels[i]] += 1

    sum_a = np.sum(matrix, axis=1)
    sum_b = np.sum(matrix, axis=0)
    total = np.sum(matrix)
    sum_ab = np.sum(sum_a * (sum_a - 1)) / 2
    sum_cd = np.sum(sum_b * (sum_b - 1)) / 2

    sum_ad_bc = np.sum(matrix * (matrix - 1)) / 2

    expected_index = sum_ab * sum_cd / total / (total - 1) + sum_ad_bc ** 2 / total / (total - 1)
    max_index = (sum_ab + sum_cd) / 2
    return (sum_ad_bc - expected_index) / (max_index - expected_index)

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']
    
    # Similarity Matrix
    num_samples = dataset.shape[0]
    sim_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            sim_matrix[i, j] = gaussian_similarity(dataset[i], dataset[j], sigma)
    
    #Laplacian MAtrix
    degree_matrix = np.diag(np.sum(sim_matrix, axis=1))
    laplacian = degree_matrix - sim_matrix
    
    eigenvectors,eigenvalues = eigh(lapalcian)
    _, computed_labels = kmeans2(eigenvectors[:, 1:k], k, minit='++')
    SSE = calculate_SSE (dataset , computed_labels)
    ARI = compute_adjusted_rand_index(true_labels, computed_labels)
    
    return computed_labels, SSE, ARI, eigenvalues

def perform_spectral_clustering_analysis(dataset, true_labels):
    """
    Conduct a hyperparameter analysis for spectral clustering on the provided dataset,
    evaluating the performance across a range of sigma values.

    Parameters:
    - dataset: A numpy array of the input data with shape (n_samples, n_features).
    - true_labels: A numpy array of true labels for the dataset.

    Returns:
    - Tuple containing:
        - An array of sigma values tested.
        - An array of ARI scores corresponding to each sigma value.
        - An array of SSE scores corresponding to each sigma value.
    """
    adjusted_rand_indices = []
    sum_of_squared_errors = []

    sigma_values = np.logspace(-1, 1, num=10)  
    num_clusters = 5  

    for sigma in sigma_values:
        _, sse, ari, _ = spectral_clustering_evaluation(sample_data, sample_labels, {'sigma': sigma, 'k': num_clusters})
        sum_of_squared_errors.append(sse)
        adjusted_rand_indices.append(ari)

    return sigma_values, np.array(adjusted_rand_indices), np.array(sum_of_squared_errors)



def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')

    data_subset = cluster_data[:1000]
    labels_subset = cluster_labels[:1000]

    sigma_range, ari_results, sse_results = spectral_hyperparameter_study(data_subset, labels_subset)

    pdf_output = pdf.PdfPages("spectral_clustering_results.pdf")

    # Plot ARI scores against sigma values
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_range, ari_results, marker='o', color='r')
    plt.title('ARI Scores vs Sigma Values')
    plt.xlabel('Sigma')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.grid(True)
    plt.xscale('log') 
    pdf_output.savefig() 
    plt.close()

    # Plot SSE scores against sigma values
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_range, sse_results, marker='o', color='b')
    plt.title('SSE Scores vs Sigma Values')
    plt.xlabel('Sigma')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.xscale('log') 
    pdf_output.savefig() 
    plt.close()

    # Assuming a function 'spectral_clustering_evaluation' is defined elsewhere
    best_sigma = 0.1
    cluster_count = 5
    analysis_results = {}

    # Apply best hyperparameters on five slices of data
    for slice_index in range(5):
        slice_data = cluster_data[slice_index * 1000: (slice_index + 1) * 1000]
        slice_labels = cluster_labels[slice_index * 1000: (slice_index + 1) * 1000]
        labels_pred, sse, ari, eigenvalues = spectral_clustering_evaluation(slice_data, slice_labels, {'sigma': best_sigma, 'k': cluster_count})
        analysis_results[slice_index] = {"labels": labels_pred, "SSE": sse, "ARI": ari, "eigenvalues": eigenvalues}

    highest_ari_slice = max(analysis_results, key=lambda x: analysis_results[x]['ARI'])

    # Plot the clusters for the dataset with the highest ARI
    plt.figure(figsize=(8, 6))
    plt.scatter(slice_data[:, 0], slice_data[:, 1], c=analysis_results[highest_ari_slice]["labels"], cmap='viridis')
    plt.title(f'Best Clustering Result (Highest ARI) - Slice {highest_ari_slice+1}')
    plt.colorbar(label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    pdf_output.savefig() 
    plt.close()

    # Find the dataset with the lowest SSE
    lowest_sse_slice = min(analysis_results, key=lambda x: analysis_results[x]['SSE'])

    # Plot the clusters for the dataset with the lowest SSE
    plt.figure(figsize=(8, 6))
    plt.scatter(slice_data[:, 0], slice_data[:, 1], c=analysis_results[lowest_sse_slice]["labels"], cmap='viridis')
    plt.title(f'Best Clustering Result (Lowest SSE) - Slice {lowest_sse_slice+1}')
    plt.colorbar(label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    pdf_output.savefig() 
    plt.close()

    # Plot of the eigenvalues for all datasets
    plt.figure(figsize=(8, 6))
    for slice_index, data in analysis_results.items():
        plt.plot(np.sort(data["eigenvalues"]), label=f'Slice {slice_index+1}')

    plt.title('Eigenvalues Plot for All Slices')
    plt.suptitle('Spectral Clustering')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    pdf_output.savefig() 
    plt.close()

    pdf_output.close()

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    ari_scores = [result["ARI"] for result in analysis_results.values()]
    average_ari = np.mean(ari_scores)
    standard_deviation_ari = np.std(ari_scores)

    # A single float
    answers["mean_ARIs"] = average_ari 

    # A single float
    answers["std_ARIs"] = standard_deviation_ari 
    
    # Extract SSE values from the analysis results and compute statistics
    sse_scores = [result["SSE"] for result in analysis_results.values()]
    average_sse = np.mean(sse_scores)
    standard_deviation_sse = np.std(sse_scores)

    # A single float
    answers["mean_SSEs"] = average_sse 

    # A single float
    answers["std_SSEs"] = standard_deviation_sse 

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
