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
from Config import Config
import hgcal2DPlot as  hplt

def load(fObjName, suffix=''):
    file_ = open( fObjName +"-"+suffix+".obj", 'rb')
    obj = pickle.load( file_)
    return obj

if __name__ == "__main__":
    c = Config()
    obj = load( c.prefixObjNameEvaluation, suffix=str(c.nValidation) )
    print ('file size', len(obj), len(obj[0]))
    for i in range(len(obj[0])):
      hplt.plotAnalyseDataSet ( obj, i )
