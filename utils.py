import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import skdim
from random import shuffle
from PIL import Image
import imagehash
import cmath


def load_dataset(directory, file_format='.npy', load_pose=True, pose_filename='pose.npy'):
    '''
    Loads all .npy or .jpg images and the pose data per image from a given directory.

    Args:
        directory (str): path to the images to be loaded.
        file_format (str): format of the images. Accepted formats are .npy and .jpg.
        load_pose (bool): if True, it will also load the pose data.
        pose_filename (str): name of the file with the pose data. The accepted format is .npy.

    Returns:
        images (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) and normalized values [0,1].
        pose (2D numpy array): pose data with (x,y) coordinates and angle (in degrees; [0,360]), wit shape (n_samples, 3).
    '''
    ## Load images.
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(file_format) and filename != pose_filename:
            filepath = os.path.join(directory, filename)
            if file_format == '.npy':
                image = np.load(filepath)
            elif file_format == '.jpg':
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    images = np.array(images)

    if np.max(images) > 1:   # normalize to [0,1] if values are RGB [0,255].
        images = images/255.

    ## Load pose (position and orientation).
    pose = []
    if load_pose:
        pose = np.load(directory+'/'+pose_filename)

    return images, pose


def load_pose_data_from_csv(path):
    '''
    Loads the pose and orientation of the image dataset from a .csv file.

    Args:
        path (str): path to the .csv file containing the position 'X', 'Y', and orientation 'Z'.

    Returns:
        position (2D numpy array): pose data with shape (n_samples, x, y), and normalized values [0,1].
        orientation (2D numpy array): orientation data with shape (n_samples, vec_x, vec_y) and angle values in vectorial form.
        angles (2D numpy array): orientation data with shape (n_samples, theta) and values in radians [0,2pi], where 0 (or 2pi) 
                                 is west and pi/2 is south.
        conjuctive (2D numpy array): pose and orientation data with shape (n_samples, x, y, vec_x, vec_y).
    '''
    data = pd.read_csv(path)

    Xpose = data['X']
    Xpose = Xpose - Xpose.min()
    Ypose = data['Y']
    Ypose = Ypose - Ypose.min()
    position = np.vstack((Xpose, Ypose)).T
    position = position / position.max()

    angles = data['Z'] + np.pi
    n_samples = angles.shape[0]
    orientation = np.zeros((n_samples, 2))
    for i in range(n_samples):
        orientation[i] = [np.cos(angles[i]), np.sin(angles[i])]

    conjunctive = np.hstack((position, orientation))

    return position, orientation, angles, conjuctive


def shuffle_2D_matrix(m):
    '''
    Shuffles a matrix across both axis (not only the first axis like numpy.permutation() or random.shuffle()).

    Args:
        m (2D numpy array): 2D matrix with arbitrary values.

    Returns:
        m_shuffled (2D numpy array): the original matrix 'm', with all the elements shuffled randomly.
    '''
    N = m.size
    ind_shuffled = np.arange(N)
    shuffle(ind_shuffled)
    ind_shuffled = ind_shuffled.reshape((m.shape[0], m.shape[1]))
    ind_x = (ind_shuffled/m.shape[1]).astype(np.int_)
    ind_y = (ind_shuffled%m.shape[1]).astype(np.int_)
    m_shuffled = m[ind_x, ind_y]
    return m_shuffled


def linear_decoding_score(embeddings, features, n_baseline=10000):
    '''
    Computes the score of linear regression of embeddings --> features. Features will normally be position (x,y) 
    or orientation (radians or in vectorial form).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        n_baseline (int; default=10000): number of permutation tests (i.e., shuffling the embeddings matrix) to compute the baseline.

    Returns:
        scores (float list): a list with two scores: (1) the evaluation of the linear regression, and (2) an average & std 
                             of n_baseline random permutation tests.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    linear_score = linear_model.score(embeddings, features)

    #baseline
    baselines = []
    for i in range(n_baseline):
        embeddings_shuffled = shuffle_2D_matrix(np.copy(embeddings))
        linear_model_baseline = LinearRegression()
        linear_model_baseline.fit(embeddings_shuffled, features)
        random_score = linear_model_baseline.score(embeddings_shuffled, features)
        baselines.append(random_score)

    baseline_score = [np.mean(baselines), np.std(baselines)]

    ratio = linear_score/(baseline_score[0])

    return linear_score, baseline_score, ratio


def ratemap_filtered_Gaussian(ratemap, std=2):
    '''
    Adds Gaussians filters to a ratemap in order to make it more spatially smooth.

    Args:
        ratemap (2D numpy array): unfiltered ratemap with the activity counts across space.
        std (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units). 

    Returns:
        new_ratemap (2D numpy array): original ratemap filtered with Gaussian smoothing.
    '''
    new_ratemap = gaussian_filter(ratemap, std)   
    return new_ratemap


def ratemaps(embeddings, position, n_bins=50, filter_width=2, occupancy_map=[], padding=False, n_bins_padding=0):
    '''
    Creates smooth ratemaps from latent embeddings (activity) and spatial position through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        occupancy_map (2D numpy array; default=[]): 2D matrix reflecting the occupancy time across the space, with shape (n_bins+2*n_bins_padding, n_bins+2*n_bins_padding).
        padding (bool; default=False): if True, 'padding_n' extra 0s are added to the walls of the arena.
        padding_n (int; default=0): the number of extra pixels that are added to every side of the arena.

    Returns:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    occ_prob = occupancy_map/np.sum(occupancy_map)

    # Add activation values to each cell in the ratemap and adds Gaussian smoothing.
    n_latent = embeddings.shape[1]
    ratemaps = np.zeros((n_latent, int(n_bins+2*n_bins_padding), int(n_bins+2*n_bins_padding)))
    for i in range(n_latent):
        ratemap_ = np.zeros((n_bins, n_bins))
        for ii, c in enumerate(embeddings[:,i]):
            indx_x = pos_imgs_norm[ii,0]
            indx_y = pos_imgs_norm[ii,1]
            #ratemaps[i, indx_x, indx_y] += c
            ratemap_[indx_x, indx_y] += c
        if padding:
            ratemaps[i] = np.pad(ratemap_, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)
        if np.any(ratemaps[i]):
            ratemaps[i] = np.abs(ratemaps[i])
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemap_filtered_Gaussian(ratemaps[i], filter_width)
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
            if len(occupancy_map) > 0:
                ratemaps[i] = ratemaps[i]/occ_prob
                ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
        
    return ratemaps


def stats_place_fields(ratemaps, peak_as_centroid=True, min_pix_cluster=0.02, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Runs a simple clustering algorithm to identify place fields, and compute their number, centroids, and sizes, for all ratemaps.

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        peak_as_centroid (bool; default=True): if True, the centroid will be taken as the peak of the place field; if False, it will take the 'center of mass'.
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        all_num_fields (1D numpy array): array with the number of place fields per embedding unit, with shape (n_latent,).
        all_centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        all_sizes (1D numpy array): array with the sizes of all place fields across embedding units, with shape (total_n_place_fields,).
    '''
    all_num_fields = []
    all_centroids = []
    all_sizes = []
    for r in ratemaps:

        ratemap = r.copy()
        
        ## Params.
        total_area = ratemap.shape[0]*ratemaps.shape[1]
        cluster_min = total_area*min_pix_cluster  #50
        cluster_max = total_area*max_pix_cluster #1250
        
        ## Clustering.
        ratemap[ratemap <  ratemap.max()*active_threshold] = 0
        ratemap[ratemap >= ratemap.max()*active_threshold] = 1

        visited_matrix  = np.zeros_like(ratemap)

        # First pass of clustering.
        clusterd_matrix = np.zeros_like(ratemap)
        current_cluster = 1

        # go through every bin in the ratemap.
        for yy in range(1,ratemap.shape[0]-1):
            for xx in range(1,ratemap.shape[1]-1):
                if ratemap[  yy, xx ] == 1:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                clusterd_matrix[ yy,xx ] = clusterd_matrix[ yy+ty, xx+tx ]

                    if clusterd_matrix[ yy, xx ] == 0:
                        clusterd_matrix[ yy, xx ] = current_cluster
                        current_cluster += 1
                        
        # Refine clustering: neighbour bins to same cluster number.
        for yy in range(1,clusterd_matrix.shape[0]-1):
            for xx in range(1,clusterd_matrix.shape[1]-1):
                if clusterd_matrix[  yy, xx ] != 0:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                if clusterd_matrix[ yy+ty, xx+tx ] != clusterd_matrix[  yy, xx ]:
                                    clusterd_matrix[ yy+ty, xx+tx ] = clusterd_matrix[  yy, xx ]
                  
        ## Quantify number of place fields.
        clusters_labels = np.delete(np.unique(clusterd_matrix), np.where(  np.unique(clusterd_matrix) == 0 ) )
        n_place_fields_counter = 0
        clusterd_matrix_ = np.copy(clusterd_matrix)
        clusters_labels_ = np.copy(clusters_labels)
        for k in range(clusters_labels.size):
            n_bins = np.where(clusterd_matrix == clusters_labels[k])[0].size
            if cluster_min <= n_bins <= cluster_max:
                n_place_fields_counter += 1
            else:
                clusterd_matrix_[np.where(clusterd_matrix_==clusters_labels[k])] = 0
                clusters_labels_ = np.delete(clusters_labels_, np.where(clusters_labels_ == clusters_labels[k]) )

        all_num_fields.append(n_place_fields_counter)
        
        ## Compute centroids.
        centroids = []
        for k in clusters_labels_:
            if peak_as_centroid:  # compute centroid as the peak of the place field.
                x, y = np.unravel_index(np.argmax( r * (clusterd_matrix_==k) ), r.shape)
                #x = np.argmax(  r * (clusterd_matrix_==k) ) 
                #y = np.argmax(  r * (clusterd_matrix_==k) )
            else:  # compute the centroid as weighted sum ('center of mass').
                w_x = r[np.where(clusterd_matrix_==k)[0], :].sum(axis=1)
                w_x = w_x/w_x.sum()
                x = np.sum(w_x * np.where(clusterd_matrix_==k)[0])
                
                w_y = r[:, np.where(clusterd_matrix_==k)[1]].sum(axis=0)
                w_y = w_y/w_y.sum()
                y = np.sum(w_y * np.where(clusterd_matrix_==k)[1])
            centroids.append([x,y])

        all_centroids += centroids
        
        ## Compute sizes of place fields.
        sizes = []
        for k in clusters_labels_:
            n_bins = np.where(clusterd_matrix_ == k)[0].size
            sizes.append(n_bins)

        all_sizes += sizes
    
    return np.array(all_num_fields), np.array(all_centroids), np.array(all_sizes)


def polarmaps(embeddings, angles, n_bins=20):
    '''
    Creates polarmaps from embedding activity and angle orientation through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).
        n_bins (int; default=20): resolution of the discretization of angles from which the polarmaps will be computed.

    Returns:
        polarmaps (2D numpy array): 2D matrix containing the polarmaps associated to all embedding units, with 
                                    shape (n_latent, n_bins).
    '''
    # Normalize orientation with respect to resolution to convert orientation to polarmap indices
    orien_imgs_norm = np.copy(angles)
    orien_imgs_norm = orien_imgs_norm/np.max(orien_imgs_norm)
    orien_imgs_norm = orien_imgs_norm*n_bins
    orien_imgs_norm = orien_imgs_norm.astype(int)
    
    # Add activation values to each cell in the ratemap and adds Gaussian smoothing
    n_latent = embeddings.shape[1]
    polarmaps = np.zeros((n_latent, n_bins))
    for i in range(n_latent):
        for ii, c in enumerate(embeddings[:,i]):
            indx = orien_imgs_norm[ii]
            polarmaps[i, indx-1] += c
        if np.any(polarmaps[i]):
            polarmaps[i] = polarmaps[i]/np.max(polarmaps[i])
        
    return polarmaps


def clean_embeddings(embeddings, normalize=False):
    '''
    Takes only the embeddings that are active an any given point, and, if normalize=True, normalizes the values.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        normalize (bool; default=False): if True, the embedding activation values will be normalized to [-1,1], per each unit.

    Returns:
        embeddings_clean (2D numpy array): original embeddings matrix, with the silent units removed, with shape (n_samples, n_active).
    '''
    indxs_active = np.any(embeddings, axis=0)
    n_active = np.sum(indxs_active)
    embeddings_clean = embeddings[:,indxs_active]

    if normalize:  # normalize between [-1, 1].
        n_samples = embeddings.shape[0]
        maxs = np.tile( np.abs(embeddings_clean).max(axis=0), n_samples).reshape((n_samples, n_active))
        embeddings_clean = embeddings_clean / maxs

    return embeddings_clean


def clean_ratemaps(ratemaps):
    '''
    Discards the ratemaps of the silent units.
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
    Returns:
        ratemaps_clean (3D numpy array): 3D matrix containing the ratemaps associated to the embedding units that are not silent, with shape (n_latent-n_silent, n_bins, n_bins).
    '''
    indxs_active = np.any(ratemaps, axis=0)
    ratemaps_clean = ratemaps[indxs_active]

    return ratemaps_clean


def angular_distance(angle1, angle2):
    '''
    Computes the angular distance between two angles given in radians.

    Args:
        angle1, angle2 (float): angles to compare (in radians).

    Returns:
        dist (float): distance value between angle1 and angle2.
    '''
    delta = np.abs(angle1 - angle2)
    dist = np.min([delta, 2*np.pi - delta])

    return dist


def euclidean_distance(point1, point2):
    '''
    Computes the Euclidean distance (L2 norm) between two points in space.

    Args:
        point1, point2 (1D numpy array): array representing a point in am arbitrary N-dimensional space.

    Returns:
        dist (float): Euclidean distance between point1 and point2 in the N-dimensional space.
    '''
    dist = np.sqrt(np.sum((point1 - point2)**2))

    return dist


def hashing_data(dataset, in_bits=True):
    '''
    Performs perceptual hashing over all images in the dataset.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        in_bits (bool; default=True): if True, the hashes will be converted to 64 bits; if False, the hashes will be returned as str.

    Returns:
        hashes (1D or 2D numpy array): 1D array with all the hashing values for each image in 'hash' or str format with shape (n_samples,) 
                                       if in_bits=False, or a 2D array with all the hashing values in bits, with shape (n_samples, 64) if in_bits=True.
    '''
    dataset_ = np.copy(dataset)

    if dataset_.shape[1] <= 3:  # if channels are in the second dimension, bring them to the last.
        dataset_ = np.transpose(dataset_, (0,2,3,1))

    if np.max(dataset_) <= 1:
        dataset_ *= 255

    dataset_ = np.uint8(dataset_)

    hashes = np.array([imagehash.phash(Image.fromarray(img)) for img in dataset_])

    n_samples = dataset.shape[0]
    if in_bits:
        hashes = np.array([np.array(hashes[i].hash.flatten()).astype(np.int_).tolist() for i in range(n_samples)])

    return hashes


def hamming_dist_matrix(hashes):
    '''
    Computes Hamming distances between hashes, as well as a corresponding similarity matrix.

    Args:
        hashes (1D or 2D numpy array): 1D array with all the hashing values for each image in 'hash' or str format with shape (n_samples,),
                                       or a 2D array with all the hashing values in bits, with shape (n_samples, 64).

    Returns:
        hash_diffs (2D numpy array): square matrix with the pairwise Hamming distances between all hashes, with shape (n_samples, n_samples).
        similarity_matrix (2D numpy array): square matrix with the pairwise normalized similarity score based on the hash_diffs, with shape (n_samples. n_samples).
    '''
    n_samples = hashes.shape[0]

    hash_diffs = []
    if len(hashes.shape) == 1:  # if hashes are in str values.
        hash_diffs = hashes.reshape((1, n_samples)) - hashes.reshape((n_samples, 1))
    else:  # otherwise they will be in 64-bit values.
        hash_diffs = np.sum(hashes[:, np.newaxis, :] != hashes, axis=2)

    #hash_diffs_norm = hash_diffs / hash_diffs.max()
    #similarity_matrix = 1. - hash_diffs_norm

    similarity_matrix = -hash_diffs

    return hash_diffs, similarity_matrix


def occupancy_map(position, n_bins=50, filter_width=2, padding=False, n_bins_padding=0):
    '''
    Computes the occupancy map based on the position through time.

    Args:
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        padding (bool; default=False): if True, 'padding_n' extra 0s are added to the walls of the arena.
        padding_n (int; default=0): the number of extra pixels that are added to every side of the arena.

    Returns:
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    map_occ = np.zeros((n_bins, n_bins))
    for p in pos_imgs_norm:
        ind_x, ind_y = p
        map_occ[ind_x, ind_y] += 1

    if padding:
        map_occ = np.pad(map_occ, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)

    map_occ = ratemap_filtered_Gaussian(map_occ, filter_width)
    map_occ = map_occ/np.max(map_occ)
    occupancy_map = map_occ.T

    return occupancy_map


def spatial_information(ratemaps, occupancy_map):
    '''
    Spatial information score (SI) as computed in Skaggs et al. 1996. The SI is computed per rate (i.e., embedding unit).

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        SI (1D numpy array): array with SI scores, in bit/spike, with shape (n_latent,).
    '''
    ratemaps_ = ratemaps[np.any(ratemaps, axis=(1,2))]
    FR = ratemaps_/(np.mean(ratemaps_, axis=(1,2))[:,np.newaxis,np.newaxis])
    OT = occupancy_map/np.sum(occupancy_map)
    log_FR = np.log2(FR, out=np.zeros_like(FR, dtype='float64'), where=(FR!=0))
    SI = np.sum(FR*OT*log_FR, axis=(1,2))
    
    return SI


def homogeneity_2Dtiling(centroids):
    '''
    Computes a metric of how homogeneously the 2D-space is tiled by the place fields, based on the coefficient of variation of
    the pair-wise minimum distances between centroids (the closer to 0 the better): 
    for all i -> Std( min( d(pos[i], pos[j..k]) ) ) / Mean( min( d(pos[i], pos[j..k]) ) ).

    Args:
        centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).

    Returns:
        cv (float): coefficient of variation of minimum distances across centroids. Values close to 0 indicate homogeneity.
    '''
    n_centroids = centroids.shape[0]
    centroid_dists = []
    for i in range(n_centroids):
        centroid_dists_all = []
        for j in range(i, n_centroids):
            if i != j:
                centroid_dist = euclidean_distance(centroids[i], centroids[j])
                centroid_dists_all.append(centroid_dist)
        if i != j:
            centroid_dists.append( np.min(centroid_dists_all) )
    cv = np.std(centroid_dists) / np.mean(centroid_dists)
    return cv


def dist_to_walls(centroids, occupancy_map):
    '''
    Computes the minimum distance to the arena's walls for every place field centroid.

    Args:
        centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        min_distances (1D numpy array): array with the distance to the nearest wall for every centroid, with shape (total_n_place_fields,).
    '''
    maze = np.copy(occupancy_map)
    maze[ occupancy_map <  occupancy_map.max()*0.1 ] = 0
    maze[ occupancy_map >= occupancy_map.max()*0.1 ] = 1
    maze = maze.T

    # Add padding to the maze and adjust centroids coordinates.
    padded_maze = np.pad(maze, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    adjusted_centroids = centroids + np.array([1, 1])

    # Get the coordinates of all the 0s in the padded maze matrix.
    wall_coords = np.argwhere(padded_maze == 0)
    
    # Compute the Euclidean distances between each adjusted centroid and all the 1s in the padded maze.
    distances = cdist(adjusted_centroids, wall_coords)
    
    # Take the minimum distance for each centroid
    min_distances = np.min(distances, axis=1)
    
    return min_distances


def angles_to_vec(angles):
    '''
    Transforms angles in radians to vectors along the unit-circle.

    Args:
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).

    Returns:
        orientation_vec (2D numpy array): 2D array with the angles converted to vectors in the 2D space, with space (n_samples, 2). 
    '''
    angles_rad = np.copy(angles)
    if np.max(angles) > 2*np.pi:
        angles_rad = np.radians(angles)

    n_samples = angles_rad.shape[0]
    orientation_vec = np.zeros((n_samples, 2))
    for i in range(n_samples):
        orientation_vec[i] = [np.cos(angles_rad[i]), np.sin(angles_rad[i])]

    return orientation_vec


def intrinsic_dimensionality(dataset, method='PCA'):
    '''
    Computes the intrinsic dimensionality of a dataset based on the scikit-dim library. Supported methods are PCA-based and MLE (as in Levina & Bickel (2004)).
    
    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        method (str; default='PCA'): method to estimate the intrinsic dimensionality of the dataset. Valid arguments are 'PCA', and 'MLE', 'TwoNN', and 'FisherS'.

    Returns:
        D (float): intrinsic dimensionality.
    '''
    N = dataset.shape[0]
    X = np.reshape(dataset, (N, -1))

    estimator_D = None
    if method == 'PCA':
        estimator_D = skdim.id.lPCA()
    elif method == 'MLE':
        estimator_D = skdim.id.MLE()
    elif method == 'TwoNN':
        estimator_D = skdim.id.TwoNN()
    elif method == 'FisherS':
        estimator_D = skdim.id.FisherS()
    else:
        print('The specific method is not supported, please check the supported methods in the documentation.')

    D = estimator_D.fit_transform(X)

    return D


def input_output_similarity(dataset, embeddings, N=1e5):
    '''
    Estimates the slope of the correlation between image and embedding similarity, to check whether similar images lead to similar embeddings.
    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        N (float; default=1e5): number of random pairs of samples from the dataset and embeddings used to compute the correlation.
    Returns:
        corr_score (float): slope of the spearman's correlation (rank-based) between image similarity and embedding similariy, within the range [0,1].
        slope (float): slope of the linear fit between pairwise dataset distances and embedding distances.
    '''
    spatial_pos_dist = []
    latent_vec_dist = []
    for i in range(int(N)):
        ind_1, ind_2 = np.random.choice(np.arange(dataset.shape[0]), 2, replace=False)
        spatial_pos_dist.append( euclidean_distance(dataset[ind_1], dataset[ind_2]) )
        latent_vec_dist.append( euclidean_distance(embeddings[ind_1], embeddings[ind_2]) )

    slope, _ = np.polyfit(spatial_pos_dist, latent_vec_dist, 1)

    corr_score = spearmanr(spatial_pos_dist, latent_vec_dist).correlation.round(2)

    return corr_score, slope


def population_sparseness(ratemaps, active_threshold=0.2):
    '''
    Estimates the population sparseness as the expected number of active units per pixel (i.e., location in space).
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].
    Returns:
        sparseness (float): population sparseness score as 1 minus the average proportion of active units across the environment, within the range [0,1].
    '''
    ratemaps_thres = np.copy(ratemaps)
    ratemaps_thres[ratemaps_thres<active_threshold] = 0
    ratemaps_thres[ratemaps_thres>=active_threshold] = 1

    prop_active_per_pixel = np.mean(ratemaps_thres, axis=0)
    sparseness = 1 - np.mean(prop_active_per_pixel)

    return sparseness


def allocentricity(embeddings, angles, n_bins=20):
    '''
    Computes an allocentric score as the average of the circular variances (i.e., mean resultant lenghts) of the polarmaps for the non-silent units.
    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).
        n_bins (int; default=20): resolution of the discretization of angles from which the polarmaps will be computed.
    Returns:
        allocentric_score (float): average circular variance for all non-silent units.
    '''
    all_polarmaps = polarmaps(clean_embeddings(embeddings), angles, n_bins=n_bins)
    bin_centers = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    circ_vars = []
    for polarmap in all_polarmaps:
        mean_vec = np.exp(1j * bin_centers) @ polarmap / np.sum(polarmap)
        circ_var = 1 - np.abs(mean_vec)
        circ_vars.append(circ_var)

    allocentric_score = np.mean(circ_vars)

    return allocentric_score
