from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """
    # *** START YOUR CODE ***
    # Flatten image to (N = H * W, C)
    pixels = image.reshape(-1, image.shape[-1])
    # pick the num_clusters # of random indexes from H * W pixels and get the
    # rgb rows associated with those picked pixels
    idx = np.random.choice(pixels.shape[0], size=num_clusters, replace=False)
    # pick the rows associated with idx array and convert to float for centroid update step
    centroids_init = pixels[idx].astype(float)
    # *** END YOUR CODE ***
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # Initialize new_centroids array
    new_centroids = np.zeros_like(centroids)
    # Iterate max_iter times
    for i in range(max_iter):
        # Update centroids based on image and current centroids
        # Flatten image to (N, C) where N = H * W
        pixels = image.reshape(-1, image.shape[-1])
        # Compute distances from each pixel to each centroid. Details below:
        # pixels[:, np.newaxis] has shape (N, 1, C) as we are inserting new axis at index 1
        # centroids has shape (K, C), and (K, C) will be treated as (1, K, C) when we do the substraction
        # (N, 1, C) - (1, K, C) will be broadcasted to (N, K, C)
        # where each pixel in (N, 1, C) is subtracted from each centroid in (1, K, C)
        # then we take the euclidian norm along axis 2 (i.e C) to get (N, K) array of distances
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        # Find the closest centroid for each pixel
        closest_centroid_indices = np.argmin(distances, axis=1)
        # Update each centroid based on the pixels assigned to it
        for j in range(len(centroids)):
            mask = closest_centroid_indices == j
            if np.sum(mask) > 0:
                # axis=0 to compute mean ACROSS all rows assigned to this centroid
                # which means, we are averaging the R column, G column and B column separately
                # across all the selected rows to get the new centroid's RGB values
                new_centroids[j] = np.mean(pixels[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]
        # Print status update every `print_every` iterations
        if (i + 1) % print_every == 0:
            print(f'Iteration {i + 1} completed')
    # *** END YOUR CODE ***
    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # Flatten image to (N, C) where N = H * W
    pixels = image.reshape(-1, image.shape[-1])
    # Compute distances from each pixel to each centroid
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    # Find the closest centroid for each pixel
    closest_centroid_indices = np.argmin(distances, axis=1)
    # Update each pixel with the RGB values of its closest centroid
    for i in range(len(pixels)):
        pixels[i] = centroids[closest_centroid_indices[i]]
    # Reshape back to original image shape
    image = pixels.reshape(image.shape)
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
