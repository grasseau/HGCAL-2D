#! /usr/bin/python
import numpy as np

class Config(object):

    # Detector in cm
  xMin = -266
  xMax =  266
  yMin =  273
  yMax = -273
  zMin =  320
  zMax =  520

  pidToName = dict([ (11,'e-'), (-11,'e+'), (22,'g'), (-211,'Pi-'), (211,'Pi+'), (130, 'K long') ])
  pidToLatex = dict([ (11,r'$e^{-}$'), (-11,r'$e^{+}$'), (22,r'$\gamma$'), (-211,r'$\pi^{-}$'), (211, r'$\pi^{+}$'), (130, r'$K long$') ])
  pidToIdx = dict([ (11, 0), (-11, 1), (22, 2), (-211, 3), (211, 4), (130, 5) ])
  # For classification
  pidToClass = dict([ (11,'EM'), (-11,'EM'), (22,'EM'), (-211,'Pi'), (211,'Pi'), (130, 'Pi') ])
  # Take care : Class 0 is background class
  pidToClassID = dict([ (11,1), (-11,1), (22,1), (-211, 2), (211, 2), (130, 2) ])

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
    self.histoEnergyCut = 0.5
    #
    self.plotHistogram = False
    # Pickle Output file name for training
    self.prefixObjNameTraining = "training"
    # Pickle Output file name for ecaluation
    self.prefixObjNameEvaluation = "eval"
    self.nHistoReadForTraining = 450
    self.nHistoReadForValidation = 50

    self.nTraining    = 1000
    self.nValidation = 50
    self.minObjectsPerEvent = 2
    self.maxObjectsPerEvent = 8