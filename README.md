# K-means Clusterning
The K-means algorithm is a method to automatically cluster similar data points together.

Concretely, we are given a training set ${(x^1,…….,x^m)}$, and we want to group the data into a few cohesive “clusters”.
K-means is an iterative procedure that starts by guessing the initial centroids, and then Refines this guess by

Repeatedly assigning examples to their closest centroids, and then
Recomputing the centroids based on the assignments.
In pseudocode, the K-means algorithm is as follows:
```python
# Initialize centroids
# K is the number of clusters
centroids = kMeans_init_centroids(X, K)

for iter in range(iterations):
    # Cluster assignment step: 
    # Assign each data point to the closest centroid. 
    # idx[i] corresponds to the index of the centroid 
    # assigned to example i
    idx = find_closest_centroids(X, centroids)

    # Move centroid step: 
    # Compute means based on centroid assignments
    centroids = compute_centroids(X, idx, K)
```
The inner-loop of the algorithm repeatedly carries out two steps:

Assigning each training example $x^i$ to its closest centroid, and
Recomputing the mean of each centroid using the points assigned to it.
The k-means algorithm will always converge to some final set of means for the centroids.

However, the converged solution may not always be ideal and depends on the initial setting of the centroids.

Therefore, in practice the K-means algorithm is usually run a few times with different random initializations.
One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).
We will implement the two phases of the K-means algorithm separately in the next sections.

We will start by completing find_closest_centroid and then proceed to complete compute_centroids.

# Finding closest centroids** 
With `find_closest_centroid` function, we will compute the closet centroids.

This function takes the data matrix X and the locations of all centroids inside centroids
It should output a one-dimensional array idx (which has the same number of elements as X) that holds the index of the closest centroid (a value in ${0,...,K−1}$, where $K$ is the total number of centroids) to every training example. (Note: The index range $0$ to $K-1$ varies slightly from what is shown in the lectures (i.e. 1 to K) because Python list indices start at 0 instead of 1)
Specifically, for every example $x^i$ we set
$$c^{(i)} := j \\quad \\mathrm{that \\; minimizes} \\quad ||x^{(i)} - \\mu_j||^2$$
Where
* $c^{(i)}$ is the index of the centroid that is closest to $x^{(i)}$ (corresponds to `idx[i]` in the starter code), and
* $\\mu_j$ is the position (value) of the $j$’th centroid. (stored in `centroids` in the starter code)\n",
* $||x^{(i)} - \\mu_j||$ is the L2-norm

```python
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    # Set K
    K = centroids.shape[0]
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        dist = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            dist.append(norm_ij)
            
        idx[i] = np.argmin(dist) 
    
    return idx
```
# Computing centroid means
for every centroid $\\mu_k$ we set:
$$\\mu_k = \\frac{1}{|C_k|} \\sum_{i \\in C_k} x^{(i)}$$
Where
* $C_k$ is the set of examples that are assigned to centroid $k$
* $|C_k|$ is the number of examples in the set $C_k$
* Concretely, if two examples say $x^{(3)}$ and $x^{(5)}$ are assigned to centroid $k=2$,
    then we should update $\\mu_2 = \\frac{1}{2}(x^{(3)}+x^{(5)})$

```python
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis = 0)
        
    return centroids
```
using the above two functions we can implement k means clustering for a sample dataset. The notebook of the repository contains the implementation of k- means clustering on a sample dataset as well as compressing images with k-means. A complete tutorial is available on this medium [blog](https://hasan-shahriar.medium.com/k-means-clustering-6bbbded6a716)
