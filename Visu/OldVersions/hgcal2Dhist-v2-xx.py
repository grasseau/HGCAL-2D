#!/usr/bin/env python
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2

import dataSet as ds
import hgcal2DPlot as  hplt
    
class Config(object):

  pidToName = dict([ (11,'e-'), (-11,'e+'), (22,'g'), (-211,'Pi-'), (211,'Pi+') ])
  pidToIdx = dict([ (11, 0), (-11, 1), (22, 2), (-211, 3), (211, 4) ])
  pidToClass = dict([ (11,'EM'), (-11,'EM'), (22,'EM'), (-211,'Pi'), (211,'Pi') ])
  # Take care : Class 0 is background class
  pidToClassID = dict([ (11,1), (-11,1), (22,1), (-211, 2), (211, 2) ])

  def __init__(self):
    #fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
    #fname = "/home/llr/cms/beaudette/hgcal/samples/hgcalNtuple_electrons_photons_pions.root"
    self.fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
    #fname = "/home/llr/info/grasseau/HAhRD/Data/hgcalNtuple_electrons_photons_pions.root"

    self.TrainingName="train-ev.obj"
    # Particule Energy Cut (in GeV)
    self.pEnergyCut = 20.0
    #
    # Histogram Energy Cut (in MeV)
    # Applyed when the histogram is built
    self.hEnergyCut = 0.5
    #
    # Distribution of the particle type
    self.part = np.zeros( len(self.pidToIdx), dtype=int)
    #
    # Detector in cm
    self.xMin = -266
    self.xMax =  266
    self.yMin =  273
    self.yMax = -273
    self.zMin =  320
    self.zMax =  520

class State(Config):

  genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags',
                     'rechit_cluster2d', 'cluster2d_multicluster']
  branches  = genpart_branches
  branches += rechit_branches

  def __init__(self):
    super( State, self).__init__()
    self.curentEvID = 0
    # Open tree
    print self.__dict__

    self.tree = uproot.open(self.fname)["ana/hgc"]
    print "# entries :", self.tree.numentries
  
""" 
Describe the event 
"""
class Event(object):
  def  __init__(self):
    """ 
    Hit Bounding Box in cm
    Defaut detector size 

    self.xMin = -266
    self.xMax =  266
    self.yMin =  273
    self.yMax = -273
    self.zMin =  320
    self.zMax =  520
    """
    #
    # Hit Min, Max
    self.hXmin = + 1000
    self.hXmax = - 1000
    self.hYmin = + 1000
    self.hYmax = - 1000
    self.hZmin = + 1000
    self.hZmax = - 1000
    self.hEmax =  -10
    self.ID = -1

"""
Collect the histo2D and event features to generate 
"images", mask, bb for the training part
"""
class histo2D(object):
  # Images
  xyBins = 256
  zBins  = 256

  def __init__(self):
    
    self.h2D = []
    self.labels = []
    self.labelIDs = []
    self.ev = []
    
def _hitFilterForwardBackward( self, ev, hx, hy, hz, he):
    # Forward / Backward Filter
    if ev.forward :
      ff =  hz > 0.0
    else :
      ff =  hz < 0.0
    zz = np.absolute( hz[ ff ] )
    hee = he[ ff ]
    xx =  hx[ ff ]
    yy =  hy[ ff ]
    ev.hx = xx
    ev.hy = yy
    ev.hz = zz
    ev.he = hee
    
    ev.hXmin = np.amin( xx )
    ev.hEmax = np.amax( he )
    
    return (xx, yy, zz, he)

  def extractHistoFromEvent( self, state, ev, forward=True  ):
    """
    Input arguments 
      pid - particle ID
      he = hit enargy
      x, y, z hit positions
    Modified arguments
      histos - histo/image list
      labels - label list
      labelIDs - label ID list
    """ 
    # ??? global part
    # ??? global hXmin, hXmax, hYmin, hYmax, hZmin, hZmax, hEmax

    print ev.pid
    ll = ( pidToName[p] for p in ev.pid)
    k = pidToIdx[ ev.pid[0] ]
    print "  OK  more", ", ".join(ll), s.part[ k ]
    s.part[ k ] += 1 
    


    h1 , xedges, yedges =  np.histogram2d( xx, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    print '  len equal ?', len(xx), len(zz), len( hee )
    # norm = 255.0 / np.amax( h )
    # ??? To do in caller and remove
    h2D.append(  h  )
    labels.append( pidToClass[ pid[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pid[0]] ], dtype=np.int32) )

    h , xedges, yedges =  np.histogram2d( yy, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    # norm = 255.0 / np.amax( h )
    h2D.append( h )
    labels.append( pidToClass[ pid[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pid[0]] ], dtype=np.int32) )

  def processEvent( self, state, evID=0, eFilter=10.0,  nPartInDetect=1 ):

    print state.__dict__
    nHistos = 0
    hXmin = state.xMin
    hXmax = state.xMax
    hYmin = state.yMin
    hYmax = state.yMax
    hZmin = state.zMin
    hZmax = state.zMax
    hEmax = state.hEmax

    print
    print "<processEvent> evID =", evID
    pid = state.tree["genpart_pid"].array()[evID]
    print '  pid',pid
    r =  state.tree["genpart_reachedEE"].array()[evID]
    print "  Reached EE :", r

    #
    # -- Reached detector filter
    #
    f = (r == 2)
    pid = pid[ f ]

    e =  state.tree["genpart_energy"].array()[evID]
    e = e[ f ]
    print "  len(e)", len(e)
    print "  energy", e

    zz = state.tree["genpart_posz"].array()[evID]
    z = []
    for i in range(len(zz)):
     if ( f[i] ):
       z.append( zz[i][0] )
    #
    z = np.array( z, dtype=np.float32  )
    #
    print " len(z)", len(z)

    # -- Particle Energy filter
    #
    f = (e >= eFilter)
    # x = x[ f ]
    # y = y[ f ]
    z = z[ f ]
    e = e[ f ]
    pid = pid[ f ]

    #
    # -- Forward/Backward
    #
    f = ( z >= 0.0 )
    print "  Forward:", f
    print "  Energy: ", e
    #  -- Forward
    ef = e[ f ]
    pidf = pid[ f ]
    #  -- Backward
    f = ( z < 0.0)
    eb = e[ f ]
    pidb = pid[ f ]

    # -- Hits
    u = state.tree["rechit_x"].array()[evID]
    x = np.array( u, dtype=np.float32  )
    xmin = np.amin( x )
    hXmin = min( hXmin, xmin)
    xmax = np.amax( x )
    hXmax = min( hXmax, xmax)
    #
    u = state.tree["rechit_y"].array()[evID]
    y = np.array( u, dtype=np.float32  )
    ymin = np.amin( y )
    hYmin = min( hYmin, ymin)
    ymax = np.amax( y )
    hYmax = max( hYmax, ymax)
    #
    u = state.tree["rechit_z"].array()[evID]
    z = np.array( u, dtype=np.float32  )
    za = np.absolute( z )
    zmin = np.amin( za )
    hZmin = min( hZmin, zmin)
    zmax = np.amax( za )
    hZmax = max( hZmax, zmax)
    #
    u = state.tree["rechit_energy"].array()[evID]
    he = np.array( u, dtype=np.float32  )
    emax = np.amax( he )
    hEmax = max( hEmax, emax)

    
    print "  len(ef), len(eb)", len(ef), len(eb)
    # Forward
    if (len(ef) != 0) and ( len(ef)<= nPartInDetect):
      ev = Event()
      ev.pid = pidf
      ev.hXmin = hXmin
      ev.hXmax = hXmax
      ev.hYmin = hYmin
      ev.hYmax = hYmax
      ev.hZmin = hZmin
      ev.hZmax = hZmax
      ev.hEmax = hEmax
      ev.forward = True
      (ev.hx, ev.hy, ev.hz, ec.he) = _hitFilterForwardBackxard( ev.forward, x, y, z, he)
      extractHistoFromEvent( state, ev)
      self.ev.append(ev)
      nHistos +=2

    # Backward
    if (len(eb) != 0) and ( len(eb)<= nPartInDetect):
      ev = Event()
      ev.pid = pidf

      ev.forward = False
      extractHistoFromEvent( state, ev  )
      self.ev.append(ev)
      nHistos +=2

    return nHistos

  """
  Build "nHistos" histograms, with associated labels
  """
  def get2DHistograms ( self, state, startEventID=0, nHistos=20 ):
    i = 0
    n = 0
    #
    # If startEventID negative continue
    if (startEventID >= 0):
      curentEvID = startEventID
    #
    while i < nHistos:
       n = self.processEvent( state, evID=curentEvID, eFilter=state.pEnergyCut,  nPartInDetect=1)
       for k in range(n):
         self.append( curentEvID );
       plotAnalyseEvent ( histos[i:], labels[i:] )
       print curentEvID, n
       state.curentEvID += 1
       i += n  
  
    # plotImages( histos )
  
    print "# entries :", tree.numentries
    for p in State.pidToName.keys():
      print State.pidToName[p], " :",  part[ pidToIdx[p] ]
  
    print "hit x min/max", hXmin, hXmax
    print "hit y min/max", hYmin, hYmax
    print "hit z min/max", hZmin, hZmax
    print "hit energy max", hEmax
  
    # Postprocess & Save for training
    
    # 
    bboxes = []
    images = []
    for i in histos:
      subi, bbox = extractSubImage( i, hEnergyCut )
      # for each histo, list of bboxes
      bboxes.append( [ bbox] )
      # ??? redo extractimage ???
      h = np.where( i > hEnergyCut, i, 0.0)
      norm = 255.0 / np.amax( h )
      images.append( np.array( h * norm, dtype=np.uint8 ))
      # print "GGG ", labelIDs
  
    # print "images", images
    # print "labels", labels[0]
    # print "labelsIds", labelIDs
  
    file_ = open(fObjName, 'w')
    pickle.dump( (images, bboxes, labels, labelIDs, evIDs), file_)
  
    ### ???? return histos, labels, labelIDs



##################
###   Others   ###
##################

# Root file (branc



if __name__ == "__main__":

  s = State()
  h = histo2D()
  
  h.get2DHistograms ( s, startEventID=0, nHistos=20  )
  # getHistos ( fname, startEventID=88, nHistos=5, fObjName="eval-x.obj" )

  

