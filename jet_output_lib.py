import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
from smp_utils import *
import tokenize as tok
import re
from cms_utils import *


class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen=True, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29.,skimfilename=None):
        
        self.lumimasks = getLumiMaskRun2()
        
        # should have separate lower ptcut for gen
        self.do_gen=do_gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]  #required

        if skimfilename != None: 
            if ".root" in skimfilename: 
                self.skimfilename = skimfilename.split(".root")[0]
            else: 
                self.skimfilename = skimfilename   
        

        #self.file = uproot.recreate("jet_z_pt_data.root")
        #self.file["Events"] = {"reco_jet_pt":np.arange(500,501),"gen_jet_pt":np.arange(500,501),"z_reco_pt":np.arange(500,501),"z_gen_pt":np.arange(500,501)}
        cutflow = {}
        
        self.hists = {
            "cutflow":cutflow
        }
        self.means_stddevs = defaultdict()
    @property
    def accumulator(self):
        #return self._histos
        return self.hists

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
            
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2018' if any(re.findall(r'Test', dataset) )
               else '2016APV'   if any(re.findall(r'APV',  dataset)) 
               else '2018' if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018', dataset) ) )
               else '2017' if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017', dataset) ) )
               else '2016')
        
        #print("dataset ", dataset)
        #print("IOV ", IOV)

        #####################################
        #### Find the era from the file name
        #### Apply the good lumi mask
        #####################################
        if (self.do_gen):
            era = None
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
            lumi_mask = np.array(self.lumimasks[IOV](events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
        
        
        
        #####################################
        ### Initialize selection
        #####################################
        sel = PackedSelection()
        
        
         
        #####################################
        ### Trigger selection for data
        #####################################       
        if not self.do_gen:
            if "UL2016" in dataset: 
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele27_WPTight_Gsf | events.HLT.Photon175
            elif "UL2017" in dataset:
                trigsel = events.HLT.IsoMu27 | events.HLT.Ele35_WPTight_Gsf | events.HLT.Photon200
            elif "UL2018" in dataset:
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200
            elif "Test" in dataset: 
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200
            else:
                raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
            sel.add("trigsel", trigsel)    
            
            #print("Here is trigsel")
            #print(sel.names)
            #print(sel.require(trigsel = True))
        
        #####################################
        ### Remove events with very large gen weights (>2 sigma)
        #####################################
        if self.do_gen:
            if dataset not in self.means_stddevs : 
                average = np.average( events["LHEWeight"].originalXWGTUP )
                stddev = np.std( events["LHEWeight"].originalXWGTUP )
                self.means_stddevs[dataset] = (average, stddev)            
            average,stddev = self.means_stddevs[dataset]
            vals = (events["LHEWeight"].originalXWGTUP - average ) / stddev
            self.hists["cutflow"][dataset]["all events"] += len(events)
            events = events[ np.abs(vals) < 2 ]
            self.hists["cutflow"][dataset]["weights cut"] += len(events)

            #####################################
            ### Initialize event weight to gen weight
            #####################################
            weights = events["LHEWeight"].originalXWGTUP
        else:
            weights = np.full( len( events ), 1.0 )
        
        


        
        #####################################
        #####################################
        #####################################
        ### Gen selection
        #####################################
        #####################################
        #####################################
        if self.do_gen:
            #####################################
            ### Events with at least one gen jet
            #####################################
            sel.add("oneGenJet", 
                  ak.sum( (events.GenJet.pt > 5.) & (np.abs(events.GenJet.eta) < 2.5), axis=1 ) >= 1
            )
            events.GenJet = events.GenJet[(events.GenJet.pt > 5.) & (np.abs(events.GenJet.eta) < 2.5)]

            #####################################
            ### Make gen-level Z
            #####################################
            z_gen = get_z_gen_selection(events, sel, self.lepptcuts[0], self.lepptcuts[1] )
            z_ptcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  z_gen.pt > 20., False )
            z_mcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  (z_gen.mass > 80.) & (z_gen.mass < 110), False )
            sel.add("z_ptcut_gen", z_ptcut_gen)
            sel.add("z_mcut_gen", z_mcut_gen)

            #####################################
            ### Get Gen Jet
            #####################################
            gen_jet, z_jet_dphi_gen = get_dphi( z_gen, events.GenJet )
            z_jet_dr_gen = gen_jet.delta_r(z_gen)



            #####################################
            ### Gen event topology selection
            #####################################        
            z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
            z_pt_frac_gen = gen_jet.pt / z_gen.pt
            z_pt_asym_sel_gen =  z_pt_asym_gen < 0.3
            z_jet_dphi_sel_gen = z_jet_dphi_gen > 2.8 #np.pi * 0.5
            sel.add("z_jet_dphi_sel_gen", z_jet_dphi_sel_gen)
            sel.add("z_pt_asym_sel_gen", z_pt_asym_sel_gen)


            
            #####################################
            ### Make gen plots with Z and jet cuts
            #####################################
            kinsel_gen = sel.require(twoGen_leptons=True,oneGenJet=True,z_ptcut_gen=True,z_mcut_gen=True)
            sel.add("kinsel_gen", kinsel_gen)
            toposel_gen = sel.require(z_pt_asym_sel_gen=True,z_jet_dphi_sel_gen=True)
            sel.add("toposel_gen", toposel_gen)


            # There are None elements in these arrays when the reco_jet is not found.
            # To make "N-1" plots, we need to reduce the size and remove the Nones
            # otherwise the functions will throw exception.
            weights2 = weights[ ~ak.is_none(gen_jet) & kinsel_gen]
            z_jet_dr_gen2 = z_jet_dr_gen[ ~ak.is_none(gen_jet) & kinsel_gen]
            z_pt_asym_sel_gen2 = z_pt_asym_sel_gen[~ak.is_none(gen_jet) & kinsel_gen]
            z_pt_asym_gen2 = z_pt_asym_gen[~ak.is_none(gen_jet) & kinsel_gen]
            z_jet_dphi_gen2 = z_jet_dphi_gen[~ak.is_none(gen_jet) & kinsel_gen]
            z_pt_frac_gen2 = z_pt_frac_gen[~ak.is_none(gen_jet) & kinsel_gen]
            z_jet_dphi_sel_gen2 = z_jet_dphi_sel_gen[~ak.is_none(gen_jet) & kinsel_gen]

            # Making N-1 plots for these three

            #####################################
            ### Get gen subjets 
            #####################################
            gensubjets = events.SubGenJetAK8
            groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)

            #####################################
            ### Convenience selection that has all gen cuts
            #####################################
            allsel_gen = sel.all("kinsel_gen", "toposel_gen" )
            sel.add("allsel_gen", allsel_gen)

                        
            
        #####################################
        ### Make reco-level Z
        #####################################
        z_reco = get_z_reco_selection(events, sel, self.lepptcuts[0], self.lepptcuts[1])
        

        
        z_ptcut_reco = z_reco.pt > 20.
        z_mcut_reco = (z_reco.mass > 80.) & (z_reco.mass < 110.)
        sel.add("z_ptcut_reco", z_ptcut_reco)
        sel.add("z_mcut_reco", z_mcut_reco)
        
        #####################################
        ### Reco jet selection
        #####################################
        recojets = events.Jet[(events.Jet.pt > 20.) & (np.abs(events.Jet.eta) < 2.5)]
        sel.add("oneRecoJet", 
             ak.sum( (events.Jet.pt > 20.) & (np.abs(events.Jet.eta) < 2.5), axis=1 ) >= 1
        )
        
        #####################################
        # Find reco jet opposite the reco Z
        #####################################
        reco_jet, z_jet_dphi_reco = get_dphi( z_reco, events.Jet )
        z_jet_dr_reco = reco_jet.delta_r(z_reco)
        z_jet_dphi_reco_values = z_jet_dphi_reco
        
        #####################################
        ### Reco event topology sel
        #####################################
        z_jet_dphi_sel_reco = z_jet_dphi_reco > 2.8 #np.pi * 0.5
        z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
        z_pt_frac_reco = reco_jet.pt / z_reco.pt
        z_pt_asym_sel_reco = z_pt_asym_reco < 0.3
        sel.add("z_jet_dphi_sel_reco", z_jet_dphi_sel_reco)
        sel.add("z_pt_asym_sel_reco", z_pt_asym_sel_reco)

        kinsel_reco = sel.require(twoReco_leptons=True,oneRecoJet=True,z_ptcut_reco=True,z_mcut_reco=True)
        sel.add("kinsel_reco", kinsel_reco)
        toposel_reco = sel.require(z_pt_asym_sel_reco=True,z_jet_dphi_sel_reco=True)
        sel.add("toposel_reco", toposel_reco)

        
        # Note: Trigger is not applied in the MC, so this is 
        # applying the full gen selection here to be in sync with rivet routine
        if self.do_gen:
            presel_reco = sel.all("allsel_gen", "kinsel_reco")
        else:
            presel_reco = sel.all("trigsel", "kinsel_reco")
        allsel_reco = presel_reco & toposel_reco
        sel.add("presel_reco", presel_reco)
        sel.add("allsel_reco", allsel_reco)

    
        # There are None elements in these arrays when the reco_jet is not found.
        # To make "N-1" plots, we need to reduce the size and remove the Nones
        # otherwise the functions will throw exception.
        weights3 = weights[ ~ak.is_none(reco_jet)]
        presel_reco3 = presel_reco[~ak.is_none(reco_jet)]
        z_jet_dr_reco3 = z_jet_dr_reco[ ~ak.is_none(reco_jet)]
        z_pt_asym_sel_reco3 = z_pt_asym_sel_reco[~ak.is_none(reco_jet)]
        z_pt_asym_reco3 = z_pt_asym_reco[~ak.is_none(reco_jet)]
        z_pt_frac_reco3 = z_pt_frac_reco[~ak.is_none(reco_jet)]
        z_jet_dphi_reco3 = z_jet_dphi_reco[~ak.is_none(reco_jet)]
        z_jet_dphi_sel_reco3 = z_jet_dphi_sel_reco[~ak.is_none(reco_jet)]
        
        
        #####################################
        ### Make final selection plots here
        #####################################
        
        # For convenience, finally reduce the size of the arrays at the end
        weights = weights[allsel_reco]
        z_reco = z_reco[allsel_reco]
        reco_jet = reco_jet[allsel_reco]
        # self.hists["ptjet_mjet_u_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.mass, weight=weights )
        #self.hists["ptjet_mjet_g_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.msoftdrop, weight=weights )
        
        if self.do_gen:
            z_gen = z_gen[allsel_reco]
            gen_jet = gen_jet[allsel_reco]
            groomed_gen_jet = groomed_gen_jet[allsel_reco]
            
            if self.skimfilename != None :
                with uproot.recreate(self.skimfilename + str(time.time()) + ".root") as fout: 
                    fout["Events"] = {
                        "reco_jet": ak.zip({
                            "pt":ak.packed(ak.without_parameters(ak.fill_none(reco_jet.pt, value=np.nan))), 
                            "eta":ak.packed(ak.without_parameters(ak.fill_none(reco_jet.eta, value=np.nan))), 
                            "phi":ak.packed(ak.without_parameters(ak.fill_none(reco_jet.phi, value=np.nan))), 
                            "mass":ak.packed(ak.without_parameters(ak.fill_none(reco_jet.mass, value=np.nan))),
                        }),
                        "gen_jet": ak.zip({
                            "pt":ak.packed(ak.without_parameters(ak.fill_none(gen_jet.pt, value=np.nan))), 
                            "eta":ak.packed(ak.without_parameters(ak.fill_none(gen_jet.eta, value=np.nan))), 
                            "phi":ak.packed(ak.without_parameters(ak.fill_none(gen_jet.phi, value=np.nan))), 
                            "mass":ak.packed(ak.without_parameters(ak.fill_none(gen_jet.mass, value=np.nan))),
                        }),
                        "weights": ak.packed(ak.without_parameters(weights))
                    }

            '''
            f = open("eta_reco_jet_data.csv", "ab")
            np.savetxt(f, np.array(reco_jet.eta) )
            f.close()
            
            f = open("eta_gen_jet_data.csv","ab")
            np.savetxt(f, np.array(gen_jet.eta) )
            f.close()
            
            f = open("phi_reco_jet_data.csv","ab")
            np.savetxt(f, np.array(reco_jet.phi) )
            f.close()
            
            f = open("phi_gen_jet_data.csv","ab")
            np.savetxt(f, np.array(gen_jet.phi) )
            f.close()
            '''
            
            
        
        for name in sel.names:
            self.hists["cutflow"][dataset][name] = sel.all(name).sum()
        
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator
    
    
    
    