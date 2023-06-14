# Can grab a file on cmslpc from 
# /store/group/lpctlbsm/NanoAODJMAR_2019_V1/Production/CRAB/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/DYJetsToLLM-50TuneCUETP8M113TeV-madgraphMLM-pythia8RunIISummer16MiniAODv3-PUMoriond17_ext2-v2/190513_171710/0000/*.root
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
import glob
import pickle

from jet_output_lib import *



def response_maker_nanov9(testing=False, do_gen=True, client=None, prependstr = "root://xcache/",skimfilename=None): 

    filedir = "samples/"

    eras_data = [
        'UL16NanoAOD', 
        'UL16NanoAODAPV', 
        'UL17NanoAOD', 
        'UL18NanoAOD'
           ]
    eras_mc = [
        'UL16NanoAODv9', 
        'UL17NanoAODv9', 
        'UL18NanoAODv9'
    ]
    
    
    if not testing: 
        nworkers = 8
        chunksize = 1000000
        maxchunks = None
    else:
        nworkers = 1
        if do_gen: 
            chunksize = 1000
        else:
            chunksize=100000
        maxchunks = 3
    fileset = {}
    if not testing: 
        
        if do_gen:

            dy_mc_filestr = "DYJetsToLL_M-50_HT_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_%s_files.txt"

            for era in eras_mc: 
                filename = filedir + dy_mc_filestr % (era)
                with open(filename) as f:
                    dy_mc_files = [prependstr + i.rstrip() for i in f.readlines() if i[0] != "#" ]
                    fileset[era] = dy_mc_files
        else: 
            datasets_data = [
                'SingleElectron_UL2016APV',
                'SingleElectron_UL2016',
                'SingleElectron_UL2017',
                'EGamma_UL2018',
                'SingleMuon_UL2016APV',
                'SingleMuon_UL2016',
                'SingleMuon_UL2017',
                'SingleMuon_UL2018',
            ]

            for dataset in datasets_data: 
                filename = filedir + dataset + '_NanoAODv9_files.txt'
                with open(filename) as f:
                    data_files = [prependstr + i.rstrip() for i in f.readlines()  if i[0] != "#" ]
                    fileset[dataset] = data_files
    else: 
        if do_gen:
            fileset["Test"] = [prependstr + "/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/26793660-5D04-C24B-813E-3C1744C84D2D.root"]
        else: 
            fileset["Test"] = [prependstr + "/store/data/Run2018A/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9_GT36-v1/2820000/FF8A3CD2-3F51-7A43-B56C-7F7B7B3158E3.root"]

                    
                
    if client == None or testing == True:         
        run = processor.Runner(
            executor = processor.FuturesExecutor(compression=None, workers=nworkers),
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=maxchunks,
            skipbadfiles=True
        )
    else: 
        run = processor.Runner(
            executor = processor.DaskExecutor(client=client),
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=maxchunks,
            skipbadfiles=True
        )
        
    output = run(
        fileset,
        "Events",
        processor_instance=QJetMassProcessor(do_gen=do_gen, skimfilename=skimfilename),
    )

        
    
    if do_gen:
        fname_out = 'jet_output.pkl'
    else:
        fname_out = 'qjetmass_zjets_reco.pkl'
    with open(fname_out, "wb") as f:
        pickle.dump( output, f )
