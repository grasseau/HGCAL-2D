#!/usr/bin/env python
from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

from mayavi.mlab import *
import uproot
import pandas as pd
import numpy as np
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
# Python 3
# import concurrent.futures
import multiprocessing
ncpu = multiprocessing.cpu_count()
# Python 3
# executor = concurrent.futures.ThreadPoolExecutor(ncpu*4)

#%matplotlib inline

# Global variable
#fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_15GeV_n100.root"
fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
# Florian 
# fname = "PhotonsWithParticlesAround.root"

pidToName = [
  "n.c.", "Ga", "e+", "e-", "nu", "Mu+", "Mu-", "Pi0", "Pi+", "Pi-",
  "K0", "K+", "K-", "n", "p", "p-", "K0s", "Eta", "Lambda", "Sig+",
  "Sig0", "Sig-", "Xi0", "Xi-", "Ome-", "nBar", "LambdaBar", "Sig-Bar", "Sig0Bar", "Sig+Bar",
  "Xi0Bar", "Xi+Bar", "Ome+Bar", "", "", "", "", "", "", "",
  "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.",
  "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c."
  ]
pidToName = dict([ (11,'e-'), (-11,'e+'), (22,'Ph'), (-211,'Pi-'), (211,'Pi+'), (13,'?') ])

genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']
branches  = genpart_branches
branches += rechit_branches


import timeit
start_time = timeit.default_timer()

def readHits( tree, evID ):
  global fname, branches
  #
  # Read and cache data
  #
  cache = {}
  df = tree.pandas.df(branches, cache = cache)
  #
  rh = pd.DataFrame({name.replace('rechit_',''):df.loc[evID,name] for name in branches if 'rechit_' in name })
  print 
  print "<readHits> evID =", evID
  print "Print rechits[:5]", len(rh)
  print
  print rh.__dict__
  print len(rh.x), len(rh.y), len(rh.layer)
  return rh


def printEvent( tree, evID ):
  global fname, branches

  print
  print "<printEvent> evID =", evID  
  print "  # entries :", tree.numentries
  for name in tree.keys():
    if 'genpart_' in name:
      print "  ", tree[name]
      # x = tree[name].array() 
      # print len( x[evID] ), "-->", x[evID ]

  pid = tree["genpart_pid"].array()[evID]
  print '  pid',pid    
 
def displayGeneratedEvent( tree, evID, forward = 0 ):
  global fname, branches

  print
  print "<displayGeneratedEvent> evID =", evID, ", forward =", forward  
  print "  genpart fields :"
  for name in tree.keys():
    if 'genpart_' in name:
      print "    ", tree[name] 
      # x = tree[name].array() 
      # print len( x[evID] ), "-->", x[evID ]
  #  
  g =  tree["genpart_gen"].array()[evID]
  r =  tree["genpart_reachedEE"].array()[evID]
  x =  tree["genpart_posx"].array()[evID]
  y =  tree["genpart_posy"].array()[evID]
  z =  tree["genpart_posz"].array()[evID]
  e =  tree["genpart_energy"].array()[evID]
  pid = tree["genpart_pid"].array()[evID]
  print '  pid',pid
  for i in range( len( g )):
    if (r[i] > 1):
      xg = np.array( x[i], dtype=np.float32 )
      yg = np.array( y[i], dtype=np.float32 )
      zg = np.array( z[i], dtype=np.float32 )
      if (forward > 0):
        f = ( zg >= 0.0)
      elif (forward < 0 ) : 
        f = ( zg < 0.0)
      else:
       f = np.ones( zg.size, dtype=bool )
      #
      xg = xg[ f ]
      yg = yg[ f ]
      zg = zg[ f ]

      if ( xg.size != 0 ):
        # mlab.text3d( xg[0], yg[0], zg[0], pidToName[pid[i]])
        # mlab.text3d( xg[0], yg[0], zg[0], "xxx")
        u = mlab.points3d( xg, yg, zg, color=(1,1,1), scale_factor = 0.2)
        str = '{:.1}'.format( e[i] )
        mlab.text3d( xg[-1], yg[-1], zg[-1], str )
        str = pidToName[pid[i]]
        mlab.text3d( xg[0], yg[0], zg[0], str )
        # mlab.outline(u, name = "xxx")
   #
  # mlab.show()
  
def readEvent( tree,  evID=12 ):

  #
  # Read and cache data
  #
  cache = {}
  # GG df = tree.pandas.df(branches, cache = cache,executor=executor)
  df = tree.pandas.df(branches, cache = cache)
  #
  print "The firts events :"
  print df[:2]

  all_particles = pd.DataFrame({name.replace('genpart_',''):df.loc[evID,name] for name in branches if 'genpart_' in name })
  print "Print all_particles[:5]", len(all_particles)
  print all_particles[:5]

  
def displayHits( rHits, forward=0, norm = False, eFilter=0.0 ):

  print "<displayHits>", "forward =", forward, ", norm =", norm, ", eFilter = ", eFilter  
  print '  # of hits =', len(rHits)
  nbrHits = len(rHits)
  #
  x = np.array( rHits.x, dtype=np.float32 )
  y = np.array( rHits.y, dtype=np.float32  )
  z = np.array( rHits.z, dtype=np.float32  )
  e = np.array( rHits.energy, dtype=np.float32  )
  #
  # -- Forward/Backward
  #
  if (forward > 0):
    f = ( z >= 0.0)
  elif (forward < 0 ) : 
    f = ( z < 0.0)
  else:
    f = np.ones( z.size, dtype=bool )
  #     
  x = x[ f ]
  y = y[ f ]
  z = z[ f ]
  e = e[ f ]
  
  #
  # -- Energy filter
  #
  f = (e >= eFilter)
  x = x[ f ]
  y = y[ f ]
  z = z[ f ]
  e = e[ f ]
  print '  # of filtered hits', len(x), "/", len(rHits)
  
  #
  # -- Normalization
  #
  xmin = x.min()
  xmax = x.max()
  ymin = y.min()
  ymax = y.max()
  zmin = z.min()
  zmax = z.max()
  emin = e.min()
  emax = e.max()
  #
  print "  x min/max :", xmin, xmax
  print "  y min/max :", ymin, ymax
  print "  z min/max :", zmin, zmax
  print "  e min/max :", emin, emax
  # 
  if ( norm ):
     r = 2.0 / (xmax - xmin)
     x = (x - xmin) * r - 1.0
     r = 2.0 / (ymax - ymin)
     y = (y - ymin) * r - 1.0
     r = 2.0 / (zmax - zmin)
     z = (z - zmin) * r - 1.0
     r = 2.0 / (emax - emin)
     e = np.log(e + 1)
  #
  e = np.power(e, 0.333)

  #
  # -- Display Hits
  #
  mlab.points3d( x, y, z, e, colormap="spectral", scale_factor = 0.5 )
  # mlab.points3d( rHits.x, rHits.y, rHits.z, rHits.energy, scale_factor = 1.0 )
  # mlab.points3d( x, y, z, s, mode='cylinder', scale_factor = 1.0 )
  # mlab.show()
  mlab.colorbar()

if __name__ == "__main__":
  #
  #    Open file and define the branches
  tree = uproot.open(fname)["ana/hgc"]

  for evID in range(30, 44):
    printEvent(    tree, evID=evID)
    rh = readHits( tree, evID=evID)
  
    displayGeneratedEvent( tree, evID=evID, forward=1)
    displayHits(rh, forward=1, eFilter=0.1)
    mlab.show()

    displayGeneratedEvent( tree, evID=evID, forward=-1)
    displayHits(rh, forward=-1, eFilter=0.1)
    mlab.show()
  
