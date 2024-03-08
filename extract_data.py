import os
import numpy as np
from print_data_utils import clear
from utilities import confrim

try:
    import ROOT
except:
    print("Root is not installed. To extract Four-Vectors from a Delphes ROOT has to be installed!")

try:
    ROOT.gSystem.Load("libDelphes")
except:
    print("Delphes Library not found. Make sure you have Delphes installed and ROOT configured correctly")

def exctr_jetConstit_4_momenta(src_path:str, src_file:str, out_name:str, topBool:bool):

    PATH = "data/not_processed"

    if os.path.exists(f"{PATH}/{out_name}"):
        if confrim(f"Directory {PATH}/{out_name}/ already exists. Data might be overwritten. Proceed?"):
            print("Proceeding")
        else:
            return
    else:
        os.mkdir(f"{PATH}/{out_name}")


    chain = ROOT.TChain("Delphes")
    chain.Add(f"{src_path}/{src_file}")
    treeReader = ROOT.ExRootTreeReader(chain)

    allEntries = treeReader.GetEntries()
            
    #initialize branches
    branchParticle = treeReader.UseBranch("Particle")
    branchElectron = treeReader.UseBranch("Electron")
    branchPhoton = treeReader.UseBranch("Photon")
    branchMuon = treeReader.UseBranch("Muon")
    branchEFlowTrack = treeReader.UseBranch("EFlowTrack")
    branchEFlowPhoton = treeReader.UseBranch("EFlowPhoton")
    branchEFlowNeuturalHadron = treeReader.UseBranch("EFlowNeutralHadron")
    branchJet = treeReader.UseBranch("Jet")
    
    #loop over all entries
    #print(f"Events: {allEntries}")
    
    jets_total = 0
    jets=[]
    
    for entry in range(allEntries):
        
        treeReader.ReadEntry(entry)
        numberOfJets = branchJet.GetEntriesFast()
        
        if numberOfJets != 0:
            print(f"Jets: {branchJet.GetEntriesFast()}")
            jets_total += branchJet.GetEntriesFast()
        
        
        #loop over all jets
        for i in range(branchJet.GetEntriesFast()):
            jet = branchJet.At(i)
            jet_constit = []
            print(f"Jet P_T: {jet.PT}")
            
            constits = jet.Constituents
            
            print(f"Constituents: {constits.GetEntriesFast()}")
            #loop over constituents
            for j in range(constits.GetEntriesFast()):
                constit = jet.Constituents.At(j)                
                if constit == None:
                    continue
                elif constit.IsA() == ROOT.GenParticle.Class():
                    particle = ROOT.GenParticle(constit)
                    momentum = [particle.P4().E(), particle.P4().Px(), particle.P4().Pz(), particle.P4().Py()]
                elif constit.IsA() == ROOT.Track.Class():
                    track = ROOT.Track(constit)
                    momentum = [track.P4().E(), track.P4().Px(), track.P4().Pz(), track.P4().Py()]
                elif constit.IsA() == ROOT.Tower.Class():
                    tower = ROOT.Tower(constit)
                    momentum = [tower.P4().E(), tower.P4().Px(), tower.P4().Pz(), tower.P4().Py()]
                jet_constit.append(momentum)

            for i in range(len(jet_constit),200):
                jet_constit.append([0.0, 0.0, 0.0, 0.0])
            
            jets.append(jet_constit)
    
    print(f"Total Jets: {jets_total}")
            
    jets_np = np.array(jets)

    tags = []
    for i in range(jets_total):
        if topBool:
            tags.append([1, 0])
        else:
            tags.append([0, 1])

    print(f"Saving Data to {PATH}/{out_name}/")
    np.save(f"{PATH}/{out_name}/x_data",jets_np)
    np.save(f"{PATH}/{out_name}/y_data",tags)    