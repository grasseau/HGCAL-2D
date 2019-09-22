import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2

from Config import Config

def getbbox( histo, exclusive=False ):
   # Extraction
    sumX = histo.sum( axis=1 )
    sumY = histo.sum( axis=0 )
    lX = len(sumX)
    lY = len(sumY)
    minX = lX +1; minY = lY +1
    maxX = -1;    maxY = -1;
    for i in range(lX):
      if sumX[i] != 0:
        minX = i
        break;
    for i in range(lX-1,-1,-1):
      if sumX[i] != 0:
        maxX = i
        break;
    for i in range(lY):
      if sumY[i] != 0:
        minY = i
        break;
    for i in range(lY-1,-1,-1):
      if sumY[i] != 0:
        maxY = i
        break;

    if exclusive:
      minX = max( minX-1, 0)
      maxX = min( maxX+1, lX)
      minY = max( minY-1, 0)
      maxY = min( maxY+1, lY)

    return [ int(minX), int(maxX), int(minY), int(maxY)]

def extractSubImage( histo ):

    # Normalization
    # norm = 255.0 / np.amax( histo )
    # image = np.array( histo * norm, dtype=np.uint8 )
    image = histo
    
    (minX,  maxX, minY,  maxY) = getbbox( histo, exclusive=True)
    
    print ("  SubImage minmax: ", minX, maxX, minY, maxY)
    # print sumX[minX: maxX]
    # ???? maxX+ 1, ...
    subi = image[ minX:maxX+1, minY:maxY+1 ]
    return subi, (minX,  maxX, minY,  maxY)
  
def plotAnalyseEvent ( ev, histos, labels ):
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  from matplotlib import colors
  kernel3 =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  kernel5 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

  n =  len(histos) 
  print ("plotHistos:", len(histos), len (ev))
  print (labels)
  plt.set_cmap('nipy_spectral')
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['blue'])
  # print 'green', len(plt.get_cmap('nipy_spectral')._segmentdata['green'])
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['red'])
  """
  cmap = plt.get_cmap('nipy_spectral', 256)
  newcolors = cmap(np.linspace(0, 1, 256))
  white = np.array([1, 1, 1, 1])
  newcolors[0, :] = white
  newcmap = ListedColormap(newcolors)
  plt.set_cmap(newcmap)
  """
  # viridis = plt.get_cmap('viridis', 256)
  viridis = plt.get_cmap('nipy_spectral', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('nipy_spectral')

  # Title
  id = str( ev[0].ID )
  partName    =  Config.pidToName [ ev[0].pid[0] ]

  titles =["x-z","y-z"]
  nbrFB = len( histos) // 2
  
  for fb in range(nbrFB):
    if ev[fb].forward:
      strfb = "Forward"
    else:
      strfb = "Backward"
    plt.suptitle( "Event "+id+", "+ strfb +": " + partName, fontsize=16)

    for  xy in range(2):
        k= xy 
        i = xy + fb * 2

        plt.subplot(2, 2, k+1)
        # k = 4*i
        vmax = np.amax( histos[i] )
        print ("vmax", vmax)
        # image = np.array( histos[i] * norm, dtype=np.uint8)

        image = histos[i]
        norm = colors.Normalize(vmin=0.0, vmax=vmax )
        imgplot = plt.imshow(image, label=labels[i],vmin=0.0, vmax=vmax, norm=norm)
        #imgplot.set_norm(norm)
        plt.title( titles[xy] )
        plt.colorbar()

        zoom, frame = extractSubImage( histos[i]  )
        if (frame[0] < frame[1]) and  (frame[2] < frame[3]):
          # Zoom

          plt.subplot(2,2,k+3)
          imgplot = plt.imshow(zoom, label=labels[i], cmap=newcmap)
          imgplot.set_norm(norm)
          #  plt.patches.Rectangle( (frame[0], frame[2]), frame[1]-frame[0], frame[3]-frame[2], edgecolor='red')
          patches.Rectangle( (1,1), 40, 40, color='red', edgecolor='red')
          """
          # Mask
          plt.subplot(n,4,k+3)
          zoom = np.where( zoom > 0, vmax, 0)
          mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel3)
          imgplot = plt.imshow(mask, label=labels[i], cmap=newcmap)
          imgplot.set_norm(norm)

          plt.subplot(n,4,k+4)
          mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel5)
          imgplot = plt.imshow(mask, label=labels[i], cmap=newcmap)
          imgplot.set_norm(norm)
        """

    plt.tight_layout()
    plt.show()

def plotImages ( histos, labels ):
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  
  kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  # n =  len(histos) / 4
  n =  len(histos) 
  print ("plotHistos:", len(histos))
  print (labels)
  plt.set_cmap('nipy_spectral')
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['blue'])
  # print 'green', len(plt.get_cmap('nipy_spectral')._segmentdata['green'])
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['red'])
  """
  cmap = plt.get_cmap('nipy_spectral', 256)
  newcolors = cmap(np.linspace(0, 1, 256))
  white = np.array([1, 1, 1, 1])
  newcolors[0, :] = white
  newcmap = ListedColormap(newcolors)
  plt.set_cmap(newcmap)
  """
  viridis = plt.get_cmap('viridis', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('viridis')
  
  for i in range(n):
    plt.subplot(441)
    # k = 4*i
    k = i
    norm = 255.0 / np.amax( histos[k+0] )
    image = np.array( histos[k+0] * norm, dtype=np.uint8) 

    histos[k+0]
    imgplot = plt.imshow(image, label=labels[k+0])
    plt.colorbar()

    # Zoom
    plt.subplot(442)
    zoom, frame = extractSubImage( histos[k+0] )
    imgplot = plt.imshow(zoom, label=labels[k], cmap=newcmap)
    plt.colorbar()

    # Mask
    plt.subplot(443)
    mask = cv2.morphologyEx(zoom, cv2.MORPH_OPEN, kernel)
    imgplot = plt.imshow(mask, label=labels[k], cmap=newcmap)
    plt.colorbar()

    plt.subplot(444)
    mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel)
    imgplot = plt.imshow(mask, label=labels[k], cmap=newcmap)
    plt.colorbar()

    plt.show()


