import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict


class util_constants: 
    def __init__(self):
        self.mclabels = [ "UL16NanoAODv9", "UL17NanoAODv9", "UL18NanoAODv9"]
        lumi = [35920,41530,59740]
        self.z_xs = 6077.22
        self.lumi = dict( zip( self.mclabels, lumi ) )

class util_binning :
    '''
    Class to implement the binning schema for jet mass and pt 2d unfolding. The gen-level mass is twice as fine. 
    '''
    def __init__(self):
        self.ptreco_axis = hist.axis.Variable([30,40,50,60,70,80,90,100,120,140,160,200,500,7000], name="ptreco", label=r"p_{T,RECO} (GeV)")
        #self.ptreco_axis = hist.axis.Regular(60,0,600, name="ptreco", label=r"p_{T,RECO} (GeV)")   
        self.mreco_axis = hist.axis.Variable([0,5,10,20,40,60,80,100,150,200,250,300,350,1000], name="mreco", label=r"m_{RECO} (GeV)")
        self.ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")                
        self.mgen_axis = hist.axis.Variable( [0,2.5,5,7.5,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350,1000], name="mgen", label=r"Mass [GeV]")
        
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.lep_axis = hist.axis.StrCategory(["ee", "mm"], name="lep")
        self.n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        self.mass_axis = hist.axis.Regular(100, 0, 1000, name="mass", label=r"$m$ [GeV]")
        self.zmass_axis = hist.axis.Regular(100, 80, 100, name="mass", label=r"$m$ [GeV]")
        self.pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")                
        self.frac_axis = hist.axis.Regular(150, 0.0, 1.875, name="frac", label=r"Fraction")                
        self.dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        self.dr_fine_axis = hist.axis.Regular(150, 0, 1.5, name="dr", label=r"$\Delta R$")
        self.dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")

        
def get_z_gen_selection( events, selection, ptcut_e, ptcut_m):
    '''
    Function to get Z candidates from ee and mumu pairs in the "dressed" lepton gen collection
    '''
    isGenElectron = np.abs(events.GenDressedLepton.pdgId) == 11
    isGenMuon = np.abs(events.GenDressedLepton.pdgId) == 13
    gen_charge = ak.where( events.GenDressedLepton.pdgId > 0, +1, -1)
        
    selection.add("twoGen_ee", 
                  (ak.sum(isGenElectron, axis=1) == 2) & 
                  (ak.all(events.GenDressedLepton.pt > ptcut_e, axis=1)) & 
                  (ak.all( np.abs(events.GenDressedLepton.eta) < 2.5, axis=1)) & 
                  (ak.sum(gen_charge, axis=1) == 0)
                 )
    selection.add("twoGen_mm", 
                  (ak.sum(isGenMuon, axis=1) == 2) & 
                  (ak.all(events.GenDressedLepton.pt > ptcut_m, axis=1)) & 
                  (ak.all( np.abs(events.GenDressedLepton.eta) < 2.5, axis=1)) & 
                  (ak.sum(gen_charge, axis=1) == 0)
                 )
    selection.add("twoGen_leptons",
                  selection.all("twoGen_ee") | selection.all("twoGen_mm")
                 )
    sel = selection.all("twoGen_leptons")
    z_gen = events.GenDressedLepton.sum(axis=1)
    #z_gen = ak.where( sel, ak.sum( events.GenDressedLepton, axis=1), None )
    return z_gen
    
def get_z_reco_selection( events, selection, ptcut_e, ptcut_m):
    '''
    Function to get Z candidates from ee and mumu pairs from reconstructed leptons
    '''
    selection.add("twoReco_ee", 
                  (ak.num(events.Electron) == 2) & 
                  (ak.all(events.Electron.pt > ptcut_e, axis=1)) & 
                  (ak.all( np.abs(events.Electron.eta) < 2.5, axis=1)) & 
                  (ak.sum(events.Electron.charge, axis=1) == 0) &
                  (ak.all(events.Electron.pfRelIso03_all < 0.2, axis=1)) &
                  (ak.all(events.Electron.cutBased > 0, axis=1) )
                 )
    selection.add("twoReco_mm", 
                  (ak.num(events.Muon) == 2) & 
                  (ak.all(events.Muon.pt > ptcut_m, axis=1)) & 
                  (ak.all( np.abs(events.Muon.eta) < 2.5, axis=1)) & 
                  (ak.sum(events.Muon.charge, axis=1) == 0) &
                  (ak.all(events.Muon.pfRelIso03_all < 0.2, axis=1)) &
                  (ak.all(events.Muon.looseId > 0, axis=1))
                 )
    selection.add("twoReco_leptons",
                  selection.all("twoReco_ee") | selection.all("twoReco_mm")
                 )
    z_reco = ak.where( selection.all("twoReco_ee"), events.Electron.sum(axis=1), events.Muon.sum(axis=1) )
    return z_reco
    
def n_obj_selection( events, selection, coll, nmin=1, ptmin=120, etamax=2.5):
    '''
    Function to require at least nmin objects from events.coll that satisfy pt > ptmin and |eta| < etamax
    '''
    selection.add("oneGenJet", 
                  ak.sum( (getattr(events, coll).pt > ptmin) & (np.abs(getattr(events, coll).eta) < etamax), axis=1 ) >= nmin
                 )
        

def find_closest_dr( a, coll , verbose = False):
    '''
    Find the objects within coll that are closest to a. 
    Return it and the delta R between it and a.
    '''
    combs = ak.cartesian( (a, coll), axis=1 )
    dr = combs['0'].delta_r(combs['1'])
    dr_min = ak.singletons( ak.argmin( dr, axis=1 ) )
    sel = combs[dr_min]['1']
    return ak.firsts(sel),ak.firsts(dr[dr_min])
    
    

def get_groomed_jet( jet, subjets , verbose = False):
    '''
    Find the subjets that correspond to the given jet using delta R matching. 
    This is suboptimal, but it's hard to fix upstream. 
    '''
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    sel = dr_jet_subjets < 0.8
    total = combs[sel]['1'].sum(axis=1)
    return total, sel

    
def get_dphi( a, coll, verbose=False ):
    '''
    Find the highest-pt object in coll and return the highest pt,
    as well as the delta phi to a. 
    '''
    combs = ak.cartesian( (a, coll), axis=1 )
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts( combs['1'] ), ak.firsts(dphi)
