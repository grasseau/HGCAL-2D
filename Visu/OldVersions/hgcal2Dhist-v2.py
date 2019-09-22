#!/usr/bin/env python
"""
Process
+ The events (forward & backward) are selected to make x-z and y-z histogram
   if:
   - the particles which reached the detector is > pEnergyCut
   - then, if the # particules is<=  nPartInDetect
   Once one histogram is done, a normalized (255) is built. The noise is
   removed by applying a filter to all histogram values > hEnergyCut (h for histogram)
"""
import math
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pickle
import cv2

import dataSet as ds
from Config import Config
import hgcal2DPlot as  hplt
    

class State(Config):

  genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags',
                     'rechit_cluster2d', 'cluster2d_multicluster']
  branches  = genpart_branches
  branches += rechit_branches

  def __init__(self):
    super( State, self).__init__()
    self.currentEvID = 0
    # Distribution of the particle type (use for repporting)
    self.part = np.zeros( len(self.pidToIdx), dtype=int)
    # Open tree

    self.tree = uproot.open(self.fname)["ana/hgc"]
    print "Nbr of  entries in root file:", self.tree.numentries
    return

  def setCurrentEvID(self, evID):
      self.currentEvID = evID
      return
  
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
    self.forward = True
    self.pid = []

"""
Collect the histo2D and event features to generate 
"images", mask, bb for the training part
"""
class histo2D(object):
  # Images
  xyBins = 256
  zBins  = 256

  def __init__(self):
    # Histogram with cuff off on enery
    self.h2D = []
    self.labels = []
    self.labelIDs = []
    #  Array of arrays  [xmin, xmax, yminx, ymax]
    self.bbox = []
    self.eMax = []
    # Nbr of histogram / 2
    self.ev = []

  def clear(self):
     self.__init__()
     return

  def extractHistoFromEvent( self, state, ev, hx, hy, hz, he  ):
    """
    Build 2 2D histograms (xx//zz and yy/zz planes) from the hits
    Apply an Energy Cut on the resulting histo
    If one of the 2 histo is zero then the even must be skipped.
    Input arguments 
      pid - particle ID
      he = hit enargy
      x, y, z hit positions
    Modified arguments
      histos - histo/image list
      labels - label list
      labelIDs - label ID list
    """ 

    # Histos can be zeros because the energy cut applied to the histo
    nHistos = 0
    print " ??? ev.pid", ev.pid
    ll = ( State.pidToName[p] for p in ev.pid)
    print " ??? 2 ev.pid", ev.pid
    k = State.pidToIdx[ ev.pid[0] ]
    s.part[ k ] += 1 
    
    # Forward / Backward Filter
    if ev.forward :
      ff =  hz > 0.0
    else :
      ff =  hz < 0.0
    #
    zz = np.absolute( hz[ ff ] )
    hee = he[ ff ]
    xx =  hx[ ff ]
    yy =  hy[ ff ]


    cbins = [histo2D.xyBins, histo2D.zBins]
    crange = [[Config.xMin, Config.xMax], [Config.zMin, Config.zMax]]
    h1, xedges, yedges =  np.histogram2d( xx, zz, bins=cbins, range=crange, weights=hee )

    # print '  len equal ?', len(xx), len(zz), len( hee )
    # ??? To do in caller and remove
    h1 = np.where( h1 > state.histoEnergyCut, h1, 0.0)

    h2, xedges, yedges =  np.histogram2d( yy, zz, bins=cbins, range=crange, weights=hee )
    # norm = 255.0 / np.amax( h )
    h2 = np.where( h2 > state.histoEnergyCut, h2, 0.0)

    if ( np.amax( h1) > 0.0  and  np.amax( h2) > 0.0 ):
      self.h2D.append(  h1  )
      self.labels.append( State.pidToClass[ ev.pid[0] ] )
      self.labelIDs.append( np.array( [ State.pidToClassID[ev.pid[0]] ], dtype=np.int32) )
      self.bbox.append( np.asarray( hplt.getbbox( h1 ), dtype=np.int32) )
      self.eMax.append( np.amax( h1 ) )
      self.h2D.append( h2 )
      self.labels.append( State.pidToClass[ ev.pid[0] ] )
      self.labelIDs.append( np.array( [ State.pidToClassID[ev.pid[0]] ], dtype=np.int32) )
      self.bbox.append( np.asarray( hplt.getbbox( h2 ), dtype=np.int32) )
      self.eMax.append( np.amax( h2 ))
      nHistos +=2

    return nHistos

  def processEvent( self, state, evID=0, eFilter=10.0,  nPartInDetect=1 ):

    nHistos = 0
    #
    # Hit Min, Max
    hXmin = + 1000
    hXmax = - 1000
    hYmin = + 1000
    hYmax = - 1000
    hZmin = + 1000
    hZmax = - 1000
    hEmax =  -10
    self.ID = -1

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
    print "  energy", e

    zz = state.tree["genpart_posz"].array()[evID]
    z = []
    for i in range(len(zz)):
     if ( f[i] ):
       z.append( zz[i][0] )
    #
    z = np.array( z, dtype=np.float32  )
    #

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
      ev.forward = True
      ev.hXmin = hXmin
      ev.hXmax = hXmax
      ev.hYmin = hYmin
      ev.hYmax = hYmax
      ev.hZmin = hZmin
      ev.hZmax = hZmax
      ev.hEmax = hEmax
      ev.ID = evID
      ev.pid = pidf
      nh = self.extractHistoFromEvent( state, ev, x, y, z, he )
      if (nh > 0):
        self.ev.append(ev)
        nHistos += nh
      else:
        print "  Not selected: histogram energy cut"

    # Backward
    if (len(eb) != 0) and ( len(eb)<= nPartInDetect):
      ev = Event()
      ev.pid = pidb
      ev.forward = False
      ev.hXmin = hXmin
      ev.hXmax = hXmax
      ev.hYmin = hYmin
      ev.hYmax = hYmax
      ev.hZmin = hZmin
      ev.hZmax = hZmax
      ev.hEmax = hEmax
      ev.ID = evID
      ev.pid = pidb
      nh = self.extractHistoFromEvent( state, ev, x, y, z, he  )
      if (nh > 0):
        self.ev.append(ev)
        nHistos += nh
      else:
        print "  Not selected: histogram energy cut"

    return nHistos

  """
  Build "nHistos" histograms, with associated labels
  """
  def get2DHistograms ( self, state, startEventID=0, nRequestedHistos=20 ):
    i = 0
    n = 0
    #
    # If startEventID negative continue
    if (startEventID >= 0):
     state.currentEvID = startEventID
    #
    while i < nRequestedHistos:
       n = self.processEvent( state, evID=state.currentEvID, eFilter=state.pEnergyCut,  nPartInDetect=1)
       print "  ??? get2DHisto n histo", n
       if (n != 0 and state.plotHistogram):
         hplt.plotAnalyseEvent ( self.ev[i/2:], self.h2D[i:], self.labels[i:] )
       state.currentEvID += 1
       i += n  
  
    # plotImages( histos ) 
    for p in State.pidToName.keys():
      print State.pidToName[p], " :",  state.part[ State.pidToIdx[p] ]
  

  def save(self, prefixName, suffix):
    # 
    bboxes = []
    images = []
    IDs= []
    n = len( self.h2D)
    for i in range(n):
      h = self.h2D[i]
      # subi, bbox = extractSubImage( i, hEnergyCut )
      # for each histo, list of bboxes
      bboxes.append( [ self.bbox[i] ] )
      norm = 255.0 / np.amax( h )
      images.append( np.array( h * norm, dtype=np.uint8 ))
      # print "GGG ", labelIDs
      IDs.append( self.ev[i/2].ID)


    print "images", len(images)
    print "bboxes", len(bboxes)
    print "labels", len(self.labels)
    print "labelsIds", len(self.labelIDs)
    print "ev", len(self.ev)
  
    file_ = open( prefixName +"-"+suffix+".obj", 'w')
    pickle.dump( (images, bboxes, self.labels, self.labelIDs, IDs), file_)


class  Histo2DCompound(object):
    
  def __init__(self):
      # h2D collection []
      self.h2D = []
      # List of list of array of bbox : [ [ [xmin, ..., ymax], [xmin, ..., ymax], ... ] ] type = int32
      self.bboxes = []
      # List of list of labels
      self.labels = []
      # List of list of labelIDs
      self.labelIDs = []    
      # List of list of evID
      # One per histo 2D h2D
      self.evIDs = []

  def addHisto(self, i, h, bbox, label, labelID, evID):

      # Add the new histogram h in the Synthetic one self.h2D[i]
      self.h2D[i] = np.add( self.h2D[i], h )
      # Append the new bbox in the list
      # List of list of array of bbox : [ [ [xmin, ..., ymax], [xmin, ..., ymax], ... ] ]
      self.bboxes[i].append( bbox )
      # List of list of labels
      self.labels[i].append( label )
      # List of list of labelIDs
      self.labelIDs[i].append( labelID )
      # List of list of evID
      # One per histo 2D h2D
      self.evIDs[i].append( evID )

  def compareBboxes ( self, bboxes, bbox):
      """ Compute the max ovelapping area of 'bbox' with
      other boxes 'bboxes' """
      overlapMax = -255*255
      print "  bbox set  :", bboxes
      print "  bbox test :", bbox
      for bb in bboxes:
         xmin = max( bb[0], bbox[0] )
         xmax = min( bb[1], bbox[1] )
         xdist = xmax - xmin
         ymin = max( bb[2], bbox[2])
         ymax = min( bb[3], bbox[3] )
         ydist = ymax - ymin
         #
         area = abs(xdist * ydist)
         """ If no overlap the 'area' is negative """
         Overlap = True
         if ( xdist<0) or (ydist <0):
           Overlap = False
         if  not Overlap:
           area = - area

         print "  area, xDist, yDist",  area, xdist , ydist
        
         if overlapMax <= 0:
            # No overlap before
            if Overlap:
                overlapMax = area
            else:
                # The overlapSum is kept negative
                overlapMax = max( overlapMax, area )
         else:
            # Overlap before, overlapMax > 0
            if Overlap:
                overlapMax = max( overlapMax, area)
            # else: do nothing, keep the overlapArea value

      return overlapMax
    
  def assembleInv(self, histObj, nObjectsPerEvent = 2, maxCommonArea = 0.0):

      if (nObjectsPerEvent != 2):
          print "Not implemented"
          return

      # Partitions in 2 sets
      n = len(histObj.h2D)
      n1 = int( math.ceil( float(n)/2 ) )
      n2 = n/2

      # Insert the firsts
      self.h2D = histObj.h2D[0:n1]
      for i in range(n1):
          self.evIDs.append( [ histObj.ev[i/2].ID ] )
          self.bboxes.append( [ histObj.bbox[i] ] )
          self.labels.append( [ histObj.labels[i] ] )
          self.labelIDs.append( [ histObj.labelIDs[i] ] )

      # Add if area overlap
      print range(n1)
      print range(n1,n1+n2)
      for i in range(n1):
        for j in range(n1,n1+n2):
            cArea = self.compareBboxes( self.bboxes[i], histObj.bbox[j])
            print "cArea", cArea
            if (cArea < maxCommonArea ):
                self.addHisto( i, histObj.h2D[j],  histObj.bbox[j], histObj.labels[j], histObj.labelIDs[j], histObj.ev[j/2].ID )

  def assemble(self, state, histObj, nRequestedHistos, nObjectsPerEvent = 2, maxCommonArea = 0.0):

      n = nRequestedHistos
      # Insert and init  the firsts objects
      self.h2D = histObj.h2D[0:n]
      for i in range(n):
          # Note there are 2 histo per event (xz, yz)
          self.evIDs.append( [ histObj.ev[i/2].ID ] )
          self.bboxes.append( [ histObj.bbox[i] ] )
          self.labels.append( [ histObj.labels[i] ] )
          self.labelIDs.append( [ histObj.labelIDs[i] ] )
      print "??? assemble 1 ", len(self.h2D), n

      h = histo2D()

      # Read & append at most 'nObjectsPerEvent' objects per compoud ev.
      for i in range(n):
         addedObjs = 0
         # nbrNewObjs = np.random.randint(nObjectsPerEvent, size=1)[0] 
         nbrNewObjs =( i % nObjectsPerEvent) 
         # print "  rand / nObjectsPerEvent :", nbrNewObjs, "/", nObjectsPerEvent
         if (nbrNewObjs <1 ):
             print "  histo (break)", i,  "# of objects ", addedObjs + 1, "/", nbrNewObjs + 1
             continue
         h.clear()
         h.get2DHistograms ( state, startEventID= -1, nRequestedHistos=nbrNewObjs  )

         # for j in range( nbrObjs):
         for j in range( len(h.bbox) ):
             cArea = self.compareBboxes( self.bboxes[i], h.bbox[j] )
             print "  cArea", cArea
             if (cArea < maxCommonArea ):
                 addedObjs += 1
                 self.addHisto( i, h.h2D[j],  h.bbox[j], h.labels[j], h.labelIDs[j], h.ev[j/2].ID )
                 if (addedObjs >= nbrNewObjs):
                     break
                     
         print "  histo", i,  "# of objects ", addedObjs + 1, "/", nbrNewObjs + 1
         print

  def save(self, fObjName, suffix=''):
    #
    print "Compound.save() "
    images = []
    n = len( self.h2D )
    for i in range(n):
      h = self.h2D[i]
      print "save ", np.amin( h ), np.amax( h ), i
      # subi, bbox = extractSubImage( i, hEnergyCut )
      # for each histo, list of bboxes
      # ??? redo extractimage ???
      # h = np.where( i > hEnergyCut, i, 0.0)
      norm = 255.0 / np.amax( h )
      
      # ??? Inv x = np.array( h * norm, dtype=np.uint8 )
      images.append( np.array( h * norm, dtype=np.uint8 ))

    print "  images ", len(images)
    print "  bboxes   :", [ len(self.bboxes[o]) for o  in range(len(self.bboxes))  ]
    print "  labels   :",   [ len(self.labels[o]) for o  in range(len(self.labels))  ]
    print "  labelIds :",  [ len(self.labelIDs[o]) for o  in range(len(self.labelIDs))  ]
    print "  ev       :",    [ len(self.evIDs[o]) for o  in range(len(self.evIDs))  ]

    for e in range(len(images)):
        print "Event :", e
        #print (bbox for bbox in self.bboxes[e])
        print "  bboxes:", self.bboxes[e]
        print "  labels  :", self.labels[e]
        print "  ev ID's :", self.evIDs[e]
        print ""
    file_ = open( fObjName +"-"+suffix+".obj", 'w')
    pickle.dump( (images, self.bboxes, self.labels, self.labelIDs, self.evIDs), file_)


if __name__ == "__main__":

  np.random.RandomState(3)
  
  s = State()
  h = histo2D()

  nTraining    = 200
  nValidation = 20
  nObjectsPerEvent = 4

  print "# "
  print "#   Training : get the",  nTraining, "images with one object"
  print "# "


  #  ??? inv
  # s.plotHistogram = True
  # h.get2DHistograms ( s, startEventID=181, nRequestedHistos=nTraining  )

  h.get2DHistograms ( s, startEventID=0, nRequestedHistos=nTraining  )

  # h.save( s.prefixObjNameTraining, str(1) )

  print "# "
  print "#  Training : Start Compouned objects"
  print "# "

  o = Histo2DCompound()
  print "Current event ID :", s.currentEvID
  o.assemble( s, h, nRequestedHistos=nTraining, nObjectsPerEvent=nObjectsPerEvent, maxCommonArea = 0.0)
  o.save( s.prefixObjNameTraining, suffix=str(nTraining))
      
  print "# "
  print "#   Validation : get the",  nValidation, "images with one object"
  print "# "
  print "Current event ID :", s.currentEvID

  h.clear()
  print "???", len(h.h2D)
  h.get2DHistograms ( s, startEventID=-1, nRequestedHistos=nValidation  )
  # getHistos ( fname, startEventID=88, nHistos=5, fObjName="eval-x.obj" )

  # h.save( s.prefixObjNameTraining, str(1) )
  print "???", len(h.h2D)

  print "# "
  print "#  Validation : Start Compouned objects"
  print "# "

  o = Histo2DCompound()
  print "Current event ID :", s.currentEvID
  print "??? ass 1", len(o.h2D)
  o.assemble( s, h, nRequestedHistos=nValidation, nObjectsPerEvent=nObjectsPerEvent, maxCommonArea = 0.0)
  print "??? ass 2", len(o.h2D)
  o.save( s.prefixObjNameEvaluation, suffix=str(nValidation))
  print "Current event ID :", s.currentEvID

