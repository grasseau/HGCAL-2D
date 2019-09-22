#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 29, 2019 5:41:32 PM$"

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
import hgcal2Dhist as h2h

from Config import Config
import hgcal2DPlot as  hplt

def load(fObjName, suffix=''):
    file_ = open( fObjName +"-"+suffix+".obj", 'rb')
    obj = pickle.load( file_)
    return obj

if __name__ == "__main__":
    c = Config()
    s = h2h.State()
    h = h2h.histo2D()
    nTraining    = s.nTraining
    nValidation = s.nValidation

    s.plotHistogram = True
    # h.get2DHistograms ( s, startEventID=181, nRequestedHistos=nTraining  )

    h.get2DHistograms ( s, startEventID=132, nRequestedHistos=20  )

    for e in range(len(h.ev)):
        print( h.ev[e].pEnergy, h.sEnergyHits[e*4], h.sEnergyHits[e*4+3])