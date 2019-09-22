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
fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_15GeV_n100.root"
fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"

pidToName = [
  "n.c.", "Ga", "e+", "e-", "nu", "Mu+", "Mu-", "Pi0", "Pi+", "Pi-",
  "K0", "K+", "K-", "n", "p", "p-", "K0s", "Eta", "Lambda", "Sig+",
  "Sig0", "Sig-", "Xi0", "Xi-", "Ome-", "nBar", "LambdaBar", "Sig-Bar", "Sig0Bar", "Sig+Bar",
  "Xi0Bar", "Xi+Bar", "Ome+Bar", "", "", "", "", "", "", "",
  "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.",
  "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c.", "n.c."
  ]

import timeit
start_time = timeit.default_timer()

def readHits( evID ):
  global fname
  #
  #    Open file and define the branches
  tree = uproot.open(fname)["ana/hgc"]
  #
  branches = []
  branches += ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  #
  branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']

  #
  # Read and cache data
  #
  cache = {}
  # GG df = tree.pandas.df(branches, cache = cache,executor=executor)
  df = tree.pandas.df(branches, cache = cache)
  #
  """
  for b in branches:
    print 'branch ', b
    if 'rechit_' in b:
      print 'filtered ', b
  """

  print "The firts events :"
  print df[:2]
  rh = pd.DataFrame({name.replace('rechit_',''):df.loc[evID,name] for name in branches if 'rechit_' in name })
  print "Print rechits[:5]", len(rh)
  print
  print rh.__dict__
  print len(rh.x), len(rh.y), len(rh.layer)
  return rh


def printEvent( evID ):
  #
  #    Open file and define the branches
  fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_15GeV_n100.root"
  tree = uproot.open(fname)["ana/hgc"]
  #
  branches = []
  branches += ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  #
  branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']

  #
  # Read and cache data
  #
  #cache = {}
  ## GG df = tree.pandas.df(branches, cache = cache,executor=executor)
  #df = tree.pandas.df(branches, cache = cache)
  #
  #print "The firts events :"
  #print df[:2]

  #print tree.keys()
  print "entries :", tree.numentries
  #  genpart  = pd.DataFrame({name.replace('genpart_',''):df.loc[evID,name] for name in branches if 'genpart_' in name })
  for name in tree.keys():
    if 'genpart_' in name:
      print tree[name], len( tree[name] )
      x = tree[name].array() 
      print len( x[evID] ), "-->", x[evID ]
      
def displayGeneratedEvent( evID, forward = 0 ):
  #
  #    Open file and define the branches
  fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_15GeV_n100.root"
  tree = uproot.open(fname)["ana/hgc"]
  #
  branches = []
  branches += ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  #
  branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']

  #
  # Read and cache data
  #
  #cache = {}
  ## GG df = tree.pandas.df(branches, cache = cache,executor=executor)
  #df = tree.pandas.df(branches, cache = cache)
  #
  #print "The firts events :"
  #print df[:2]

  #print tree.keys()
  print "entries :", tree.numentries
  #  genpart  = pd.DataFrame({name.replace('genpart_',''):df.loc[evID,name] for name in branches if 'genpart_' in name })
  for name in tree.keys():
    if 'genpart_' in name:
      print tree[name], len( tree[name] )
      x = tree[name].array() 
      print len( x[evID] ), "-->", x[evID ]

  
  g =  tree["genpart_gen"].array()[evID]
  r =  tree["genpart_reachedEE"].array()[evID]
  x =  tree["genpart_posx"].array()[evID]
  y =  tree["genpart_posy"].array()[evID]
  z =  tree["genpart_posz"].array()[evID]
  e =  tree["genpart_energy"].array()[evID]

  pid =  tree["genpart_pid"].array()[evID]
  print pid

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
          # e = np.power(e, 0.333)
        # e = np.full( (xg.size),  r[i], dtype=flot32)
        mlab.text3d( xg[0], yg[0], zg[0], pidToName[pid[i]])
        u = mlab.points3d( xg, yg, zg, color=(1,1,1), scale_factor = 0.2)
        str = '{:.1}'.format( e[i] )
        print str
        mlab.text3d( xg[-1], yg[-1], zg[-1], str )
        # mlab.outline(u, name = "xxx")
   #
  # mlab.show()
  
def readEvent( evID=12):
  #
  #    Open file and define the branches
  fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_15GeV_n100.root"
  tree = uproot.open(fname)["ana/hgc"]
  #
  branches = []
  branches += ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  #
  branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']

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

def cmp_to_key(h):
   # return long( 100* (h[4] + h[5] * 1000 + h[1]*1000000) )
   # return long( 100* (h[4]*100000 + h[5] * 100 + h[1]) )
   return float(h[4])

def run1( rHits, forward=0, norm = False ):

   print 'len(hits)=', len(rHits)
   nbrHits = len(rHits)
   x = np.array( rHits.x, dtype=np.float32 )
   y = np.array( rHits.y, dtype=np.float32  )
   z = np.array( rHits.z, dtype=np.float32  )
   e = np.array( rHits.energy, dtype=np.float32  )



   if (forward > 0):
     f = ( z >= 0.0)
   elif (forward < 0 ) : 
     f = ( z < 0.0)
   else:
     f = np.ones( z.size, dtype=bool )
       
   x = x[ f ]
   y = y[ f ]
   z = z[ f ]
   e = e[ f ]

   f = (e >= 0.00)
   x = x[ f ]
   y = y[ f ]
   z = z[ f ]
   e = e[ f ]
   print '# filtered hits', len(x), "/", len(rHits)

   xmin = x.min()
   xmax = x.max()
   ymin = y.min()
   ymax = y.max()
   zmin = z.min()
   zmax = z.max()
   emin = e.min()
   emax = e.max()

   print " x min/max :", xmin, xmax
   print " y min/max :", ymin, ymax
   print " z min/max :", zmin, zmax
   print " e min/max :", emin, emax
   
   if ( norm ):
     r = 2.0 / (xmax - xmin)
     x = (x - xmin) * r - 1.0
     r = 2.0 / (ymax - ymin)
     y = (y - ymin) * r - 1.0
     r = 2.0 / (zmax - zmin)
     z = (z - zmin) * r - 1.0
     r = 2.0 / (emax - emin)
     e = np.log(e + 1)
  
   e = np.power(e, 0.333)
  

   print 'Copy finished'
   
   mlab.points3d( x, y, z, e, colormap="spectral", scale_factor = 0.5 )
   #mlab.points3d( rHits.x, rHits.y, rHits.z, rHits.energy, scale_factor = 1.0 )
   # mlab.points3d( x, y, z, s, mode='cylinder', scale_factor = 1.0 )
   # mlab.show()
   mlab.colorbar()

def run2( rHits ):

   print 'len(hits)=', len(rHits)

   nbrHits = len(rHits)
   xmin = 1000; xmax = -1;
   ymin = 1000; ymax = -1;
   zmin = 1000; zmax = -1;
   emin = 1000; emax = -1;

   nsubset = 0
   hFiltered = []
   for h in rHits:
     if ( h[4] > 0 and h[5] > 0):
       nsubset  = nsubset+1
       if ( h[4] < xmin ): xmin = h[4]
       if ( h[4] > xmax ): xmax = h[4]
       if ( h[5] < ymin ): ymin = h[5]
       if ( h[5] > ymax ): ymax = h[5]
       if ( h[1] < zmin ): zmin = h[1]
       if ( h[1] > zmax ): zmax = h[1]
       if ( h[2] < emin ): emin = h[2]
       if ( h[2] > emax ): emax = h[2]
       hFiltered.append( h )
       """
       key = (h[4],h[5])
       if key not in xyHits:
     xyHits[key] = [ [], [] ]

       xyHits[ key ][0].append( h[1] )
       xyHits[ key ][1].append( h[2] )
       """

   print 'number of selected hits: ', nsubset
   print 'x min/max: ', xmin, ', ', xmax
   print 'y min/max: ', ymin, ', ', ymax
   print 'z min/max: ', zmin, ', ', zmax
   print 'e min/max: ', emin, ', ', emax
   """
   dx = 1.0 / (xmax - xmin)
   dy = 1.0 / (ymax - ymin)
   dz = 1.0 / (zmax - zmin)
   """
   """
   0.183261 34.7569 12.0 0.00979254 85540.0
   0.183261 34.7569 12.0 0.010492 104189.0
   hit=85540, layer=25, detid=1724757657, energy=0.0188466,
Threshold(>)=0.00973913, thickness=100, x=-43.1117, y=44.0412 Ok
   hit=104189, layer=1, detid=1712166565, energy=0.0296868,
Threshold(>)=0.00613635, thickness=100, x=27.6263, y=34.0427 Ok
   hit=277, layer=38, detid=1767228605, energy=18.9633,
Threshold(>)=0.0179353, thickness=200, x=47.006, y=24.7583 Ok
   47.006 24.7583 38.0 18.9633 277.0
   hit=278, layer=3, detid=1729986627, energy=0.00668134,
Threshold(>)=0.00575439, thickness=100, x=36.2853, y=19.0449 Ok
   36.2853 19.0449 3.0 0.00668134 278.0
   """
   """ Sort """

   #for h in hits:
   #  print h[4], h[5], h[1]

   # hits = sorted( hFiltered, key=cmp_to_key )
   dtype = [('hitID', float), ('z', float), ('e', float), ('threshold',
float), ('x', float),('y', float),]
   a = np.array( hFiltered, dtype=dtype)
   hits = np.sort(a, order=['x', 'y', 'z','e'] )
   print hits[1:100]
   # hits = a
   for h in a[1:100]:
     print h[4], h[5], h[1], h[2], h[0]
   print "----------"

   for h in hits[1:100]:
     print h[4], h[5], h[1], h[2], h[0]

   nbrHits = len(hits)
   npoints = 0
   # nbrHits = 1000
   x = []; y = []; z = []; s = [];
   for hit in range(nbrHits):
      h = hits[hit]
      if ( h[2] > 0.01):
        x.append( (h[4] - xmin)); y.append( (h[5]-ymin)); z.append(
(h[1]-zmin) )
        s.append ( np.power( h[2], 0.33 ) )
        npoints = npoints + 1
      # s.append ( 1.0 )
   print 'Copy finished'
   print 'filter in energy ', 0.01 * npoints/nsubset, '% [', npoints, ']'
   # mlab.points3d( x, y, z, s, colormap="copper", scale_factor = 1.0 )
   mlab.points3d( x, y, z, s, scale_factor = 1.0 )
   # mlab.points3d( x, y, z, s, mode='cylinder', scale_factor = 1.0 )

   
if __name__ == "__main__":
    # execute only if run as a script
    printEvent(12)
    rh = readHits(12)
    #run1( rh )
    displayGeneratedEvent(12)
    run1(rh)
    mlab.show()
