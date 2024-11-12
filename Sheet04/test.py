import numpy as np
from scipy.stats import multivariate_normal as mvnormal

'''
   This code synthesizes simple 2D mixture-of-Gaussians datasets
   with which you can test your EM code.

   Use the methods testdata[123...6](). Each returns a Nx2 numpy
   array of N points.
''' 


def _generate_testdata(means, covs, ns):
    '''
    Internal method: Given GMM parameters, generate the actual dataset/points.

    The parameters are:
    - means (the clusters' centers)
    - covs (the clusters' shapes / covariance matrices)
    - ns (the number of samples to generate per cluster)
    '''
    K = len(means)
    
    samples = None
    for k in range(K):
        samplesk = mvnormal.rvs(mean=means[k],
                                cov=covs[k],
                                size=ns[k])
        samples = samplesk if samples is None else np.vstack([samples,samplesk])

    return samples



def testdata1():
    '''example from the lecture.'''
    means = [[2,4], [7,2], [7,6]]
    ns = [100, 100, 100]
    covs = [np.array([[0.5,0.8],[0.8,5]]),
              np.array([[2,0.0],[0.,0.7]]),
              np.array([[0.5,-0.3],[-0.3,0.7]])]
    return _generate_testdata(means, covs, ns)

def testdata2():
    '''different priors.'''
    means = [[2,4], [7,1], [7,8]]
    ns = [500, 100, 80]
    covs = [np.array([[0.5,0.8],[0.8,5]]),
            np.array([[2,0.0],[0.,0.7]]),
            np.array([[0.5,-0.3],[-0.3,0.7]])]
    return _generate_testdata(means, covs, ns)

def testdata3():
    '''strongly correlated data.'''
    means = [[2,4], [7,1], [7,8]]
    ns = [500, 500, 200]
    covs = [np.array([[1,0.9],[0.9,1]]),
            np.array([[2,-1.8],[-1.8,2]]),
            np.array([[0.5,-0.3],[-0.3,0.7]])]
    return _generate_testdata(means, covs, ns)

def testdata4():
    '''2 clusters.'''
    means = [[2,4], [7,1]]
    ns = [500, 500]
    covs = [np.array([[1,0.9],[0.9,1]]),
            np.array([[2,-1.8],[-1.8,2]])]
    return _generate_testdata(means, covs, ns)

def testdata5():
    '''4 clusters.'''
    means = [[2,4], [7,1], [8,7], [5,9]]
    ns = [500, 500, 500, 500]
    covs = [np.array([[1,0.9],[0.9,1]]),
            np.array([[1,0.9],[0.9,1]]),
            np.array([[3,0.],[0.,3]]),
            np.array([[2,-1.8],[-1.8,2]])]
    return _generate_testdata(means, covs, ns)

def testdata6():
    '''little training data -> instable?'''
    means = [[2,4], [7,1], [8,7]]
    ns = [10,10,10]
    covs = [np.array([[1,0.9],[0.9,1]]),
            np.array([[3,0.],[0.,3]]),
            np.array([[2,-1.8],[-1.8,2]])]
    return _generate_testdata(means, covs, ns)

