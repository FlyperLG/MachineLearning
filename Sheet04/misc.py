#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvnormal
import matplotlib.pyplot as plt
import pylab



def plot(X, model, path):
    ''' given a model (see em.py) and a dataset X, plot both.
        X is plotted as a gray point cloud, the model
        as a color map.

        THERE IS NO NEED TO CHANGE THIS METHOD (I THINK...).
    '''
    # define range to plot: range of data +- 10%.
    min_,max_ = X.min(axis=0), X.max(axis=0)
    ptp_ = max_ - min_
    min_ -= 0.1 * ptp_
    max_ += 0.1 * ptp_
    x = np.arange(min_[0], max_[0], (max_[0]-min_[0])/20.)
    y = np.arange(min_[1], max_[1], (max_[1]-min_[1])/20.)

    # plot input data X
    plt.plot(X[:,0], X[:,1], 'o',
             color='gray',
             markersize=2, 
             linewidth=4)
    plt.xlim( [x[0],x[-1]] )
    plt.ylim( [y[0],y[-1]] )

    if model is not None:

        (K,priors,means,vars) = model

        # construct a grid of 2D points over the data range
        grid = np.array(np.meshgrid(x,y)).T.reshape(-1,2)   # NPOINTS x 2

        # compute GMM density over the grid's points
        F = np.zeros(grid.shape[0])                         # NPOINS
        for p,m,v in zip(priors,means,vars):
            F += p * mvnormal.pdf(grid, mean=m, cov=np.diag(v))

        # plot GMM density
        im = plt.imshow(F.reshape([len(x),len(y)]).T,
                        interpolation='bicubic',
                        origin='lower',
                        cmap=pylab.get_cmap("RdYlGn"),
                        extent=[x[0],x[-1],y[0],y[-1]] )

    plt.grid()
    plt.savefig(path)
    plt.close()


