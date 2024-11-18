#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import misc
import test

import numpy as np
import time
import cv2
from scipy.stats import multivariate_normal as mvnormal
from sklearn.decomposition import PCA



class ExpectationMaximization:
    '''
    This class is used to train a GMM model using the EM algorithm.
    Its interface are the methods:
    - train(X, K), which runs *one* EM training, yielding a GMM with K components
    - apply(X), which returns best-matching sample from X per cluster
                (to be used in KeyframeExtractor)

    Internally, the class represents the EM model as a tuple

         (K, priors, means, vars),

    where
    - K is the number of clusters
    - priors is a K-dimensional vector of prior probabilities
    - means is a K x D matrix of cluster means
    - vars is a K x D matrix of cluster variances.
    '''

    def __init__(self, iterations=50):
        """
        class constructor.

        @type iterations: int
        @param iterations: the number of iterations to run the EM algorithm.
        """
        self.iterations = iterations

        # clip variances to at least epsilon
        self.epsilon = 0.01

        # initially, there is no model (you need to call train() first)
        self.model = None


    def _initialize(self, X, K):
        ''' initialize the model with random priors, means and variances '''

        # initialize priors
        priors = np.repeat(1. / K, K)  # K
        # initialize means
        means = X[np.random.choice(X.shape[0], K, replace=False)]  # K x D
        # initialize variances
        globalvars = X.var(axis=0)  # D
        vars = np.array(K * [globalvars])  # K x D

        return K, priors, means, vars


    def _E(self, X, model):
        '''
        the E-step. Given data X and a model, determine P(k|x,Theta),
        i.e. the probability for each datapoint x to belong to
        each cluster k.

        @rtype: np.array
        @return: the probabilities/weights w(ik) = P(k|x,Theta), in an N x K array.
        '''

        (K, priors, means, vars) = model
        result = []
        for point in X:
            probabilities = []
            for k in range(0, K):
                # Get Dimensions
                dimension = len(point)
                # Calculate variance
                variances = np.zeros((dimension, dimension))
                np.fill_diagonal(variances, vars[k])

                prob = priors[k] * mvnormal.pdf(point, means[k], variances)
                probabilities.append(prob)
            sumProbabilities = np.sum(probabilities)
            result.append(probabilities/sumProbabilities)
        
        return result
    
    def own_multivariate_normal(X, model):
        (K, priors, means, vars) = model
        result = []
        for point in X:
            probabilities = []
            for k in range(0, K):
                # Get Dimensions
                dimension = len(point)

                # Calculate variance
                variances = np.zeros((dimension, dimension))
                np.fill_diagonal(variances, vars[k])

                normalizationFactor = (1 / (((2 * np.pi) ** (dimension/2)) * (np.linalg.det(variances) ** (1/2))))
                prob = normalizationFactor * np.exp(-1/2 * np.dot(np.dot((point-means[k]).T, np.linalg.inv(variances)), (point-means[k])))
                probabilities.append(prob)
            result.append(probabilities)
        return result
                

    def _M(self, P, X, K):
        '''
        the M-step. Given the probabilites P(k|x),
        re-estimate prior, means and variances using
        the formulas from lecture.

        @rtype: tuple
        @return: a new (priors,means,vars).
        '''

        nk = np.sum(P, axis=0)
        priors = nk / len(X)

        means = []
        for k in range(0, K):
            means.append((1 / nk[k]) * np.sum([P[index][k] * point for index, point in enumerate(X)], axis=0))

        epsilon = 0.01
        vars = []
        for k in range(0, K):
            var = (1 / nk[k]) * np.sum([P[index][k] * ((point - means[k]).T * (point - means[k])) for index, point in enumerate(X)], axis=0)
            var[var < epsilon] = epsilon
            vars.append(var)
        return (priors, means, vars)

    def train(self, X, K):
        '''
        run an EM training.
        Also, stores the final model in self.model.

        @type X: np.array (NxD)
        @param X: N rows, each a D-dimensional feature vector.
        @type K: int
        @param K: the number of clusters to use.

        @rtype: tuple
        @return: the final model (tuple (K,priors,means,vars)).
        '''

        self.model = model = (K, priors, means, vars) = self._initialize(X, K)  # K x D

        for i in range(self.iterations):
            probability = self._E(X, model)
            (newPriors, newMeans, newVars) = self._M(probability, X, K)
            model = (K, newPriors, newMeans, newVars)
        self.model = model
        return model




    def _loglikelihood(self, model, X):
        '''
        computes the log-likelihood of the data X given the model.

        @type model: tuple
        @param model: the model to use (tuple (K,priors,means,vars)).
        @type X: np.array (N x D)
        @param X: the data to compute the log-likelihood for.
        '''
        (K,priors,means,vars) = model
        P = [mvnormal.pdf(X, mean=means[k], cov=np.diag(vars[k])) for k in range(K)]
        P = np.array(P).T                                          # N x K 
        P = P * priors
        P = np.maximum(P.sum(axis=1), 0.00000000001)
        logL = np.log(P).sum()
        return logL



    def train_visualize(self, X, K, plotpath="plot"):
        """
        runs an EM clustering on a given set of samples / features.

        Conducts multiple runs for K = minK, ..., maxK,
        each time calling _train_run(), and keeps the best result,
        according to the BIC score.

        @type X: np.array (NxD)
        @param X: N rows, each a D-dimensional feature vector, stacked
                  to a matrix.
        @type minK: int
        @param minK: the minimum number of clusters to try.
        @type maxK: int
        @param maxK: the maximum number of clusters to try.
        @type plotpath: string (or NoneType)
        @param plotpath: the path to save the plot to (don't plot if None).
        """
        # FIXME: implement
        self.model = model = (K, priors, means, vars) = self._initialize(X, K)  # K x D

        for i in range(self.iterations):
            probability = self._E(X, model)
            (newPriors, newMeans, newVars) = self._M(probability, X, K)
            model = (K, newPriors, newMeans, newVars)
            misc.plot(X, model, f"{plotpath}{i}")
        self.model = model
        return model


    def apply(self, X):
        """
        returns best-matching sample from X per cluster.
        You need to call train() before you can call apply().

        @type X: np.array (NxD)
        @param X: each row in X corresponds to one feature vector
        @rtype: (list, np.array (KxD))
        @return: given an input feature/line, apply() returns
                 1. the indices of the best-matching sample per cluster
                 2. a KxD-dimensional array containing the corresponding feature vectors
        """
        raise NotImplementedError() # FIXME: implement



class KeyframeExtractor:
    '''
        See last Exercise: Implement this class to extract keyframes from a video.
    '''

    def __init__(self, em, framestep=3, dpca=2):
        ''' class constructor. '''
        # use only every third frame
        self.framestep = framestep
        # an instance of ExpectationMaximization()
        self.em = em
        # project features to two dimensions
        self.dpca = dpca
        self.pca = PCA(n_components=dpca)

    def _standardize(self, X):
        ''' internal method that normalizes the features to mean=0 and std=1 '''
        means = X.mean(axis=0)
        stds = np.maximum( X.std(axis=0), 0.001 )
        return (X - means) / stds

    def _frame2feature(self, frame):
        ''' internal method that turns a video frame into a 32*32-feature vector'''
        return cv2.resize(frame, (32,32)).flatten()

    def video2features(self, filename):
        '''
        call this method to extract features from a video file (to feed into EM).
        ! DO NOT TOUCH (I THINK) !

        @type filename: string
        @param filename: the path to the video file
        @rtype: tuple
        @return: a tuple of X and imgs, where...
          - X is a NxD array, where N is the number of frames and D is the number of features
          - imgs is a list of the N original frames from the video (for visualization)
        '''
        cap = cv2.VideoCapture(filename)
        nframe = 0
        X = []
        imgs = []
        while True:
            flag, frame = cap.read()
            nframe += 1
            #print nframe, frame is None, flag
            if frame is None:
                break
            if nframe%self.framestep!=1:
                continue
            else:
                x = self._frame2feature(frame)
                X.append(x)
                imgs.append(frame)

        # refine features and project to lower dimension using PCA.
        X = np.array(X)
        X = self.pca.fit_transform(X)
        X = self._standardize(X)

        return X,imgs


    def extract_keyframes(self, filename):
        '''
        implement this method to extract a set of 'good' keyframes
        from a video file.
        '''
        raise NotImplementedError() # FIXME: implement

        
if __name__ == "__main__":

    # Exercise 1
    X = test.testdata1()
    em = ExpectationMaximization(iterations=30)
    #em.train_visualize(X, K=3, plotpath="Output/visualize")
    model = em.train(X, K=3)
    misc.plot(X, model, "Output/result")

    # Exercise 2 (video keyframes)
    '''
    em = ExpectationMaximization()
    kf = KeyframeExtractor(em, framestep=3, dpca=2)
    kf.extract_keyframes("vids/got.mp4")
    '''
