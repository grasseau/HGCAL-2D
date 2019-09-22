#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 12, 2019 3:19:29 PM$"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2

from wpca import PCA, WPCA, EMPCA

n_samples = 1000
n_features = 50

def elipsoid(axes=None, angles=None, orig=None, n=20):
    """
    axis : [Naxes] half axes ot the elipse
    angles : in rad, elipse direction or rotation
    orig :  [Naxes] shift for origin
    Return:
    - x [n, Naxes] : the sampling
    - w [n] : the weights of the points
    """

    axes = np.array(axes)
    shifts = np.array(orig)
    rots = np.array(angles)
    
    np.random.RandomState(43)
    # For normal law
    # r = np.abs( np.random.randn(n,1)  )

    r = np.random.rand(n,1)
    alpha = 2*np.pi * np.random.rand(n,1)

    # Rotation
    s = np.sin( rots[0] )
    c = np.cos( rots[0] )
    R = np.array( [ [ c , -s],
                           [s,    c] ] )
    # print(R, np.matmul(R,[0,1]))

    x = np.zeros( (n, 2) )
    x[:,0] =  np.ravel( axes[0] * r[:] * np.cos( alpha[:] ) )
    x[:,1] =  np.ravel( axes[1] * r[:] * np.sin( alpha[:] )  )

    y = np.matmul( R, x.transpose())
    x = y.transpose() + shifts
    w = np.random.rand(n,1)

    return x, w


def plotPCA( ax, pca, x ):
  means_ =  pca.mean_
  sigmas_ =  np.sqrt(pca.explained_variance_)
  vectors_ = pca.components_[:ncomp]

  ax.plot( x[:, 0], x[:, 1], 'ro' )
  a = np.arctan2( vectors_[0][1], vectors_[0][0]  )  * 180.0 / np.pi
  print("a", a)

  ax.add_patch(
    patches.Ellipse( means_, 2*sigmas_[0], 2*sigmas_[1],
                              angle=a, linewidth=2, fill=False, zorder=0
                           ) )

  ax.add_patch(
    patches.Ellipse( means_, 4*sigmas_[0], 4*sigmas_[1],
                              angle=a, linewidth=2, fill=False, zorder=-1
                            ) )
  # ax[1,1].set_title('Cluster :  axes = {0}, angle = {1}, origin {2}'.format( axes, angles[0] / np.pi , orig))
  ax.set_title('Cluster :  axes = {a}, angle = {b:4.1f} $\pi$, origin {c}'.format( a=axes, b= (angles[0] / np.pi) , c=orig), fontsize=10)

if __name__ == "__main__":
  fig, ax = plt.subplots(2, 2, figsize=(16, 6))

  ####
  # Half axes
  axes= [10.0, 1.0]
  # Rotate elispse [rad]
  angles= [np.pi*0.1]
  # Origin shift
  orig = [5.0, -3.0]
  x, w = elipsoid( axes= axes, angles= angles, orig = orig, n=400)

  ax[0, 1].plot( x[:, 0], x[:, 1], 'o' )

  # PCA
  kwds = {}
  ncomp = 2
  pca = WPCA(n_components=ncomp).fit(x, **kwds)
  Y = WPCA(n_components=ncomp).fit_reconstruct(x, **kwds)
  means_ =  pca.mean_
  sigmas_ =  np.sqrt(pca.explained_variance_)
  vectors_ = pca.components_[:ncomp]
  print("Components \n", vectors_)
  print("Sigmas", sigmas_ )
  print("Means", means_ )

 # Not used here
 # ax[1, 1].plot(np.arange(1, ncomp+1), pca.explained_variance_ratio_)


  plotPCA(  ax[1,1], pca, x)

  fig.suptitle("Test PCA, WPCA", fontsize=16)
  plt.show()
