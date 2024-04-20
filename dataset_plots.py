import os
from plot_utils import hist
from utilities import Settings
import numpy as np

# Calculate the pseudorapidity of pixel entries
def eta (x)->np.ndarray:
    pz = x[:,:,3]
    pT = get_pT(x)

    small = 1e-10
    small_pT = (np.abs(pT) < small)
    small_pz = (np.abs(pz) < small)
    not_small = ~(small_pT | small_pz)
    theta = np.arctan(pT[not_small]/pz[not_small])
    theta[theta < 0] += np.pi
    etas = np.zeros_like(pT)
    etas[small_pz] = 0
    etas[small_pT] = 1e-10
    etas[not_small] = np.log(np.tan(theta/2))
    return etas

def eta_jet(x):
    return eta(x).mean(axis=1)

def eta_cut_const(x:np.ndarray, y:np.ndarray, eta_max:float):
        etas = eta(x)
        return np.delete(x, np.where((etas <= -eta_max) | (etas >= eta_max)), axis=0), np.delete(y, np.where((etas <= -eta_max) | (etas >= eta_max)), axis=0) 

# Calculate the azimuthal angle of pixel entries
def phi (x):
    """
    phis are returned in rad., np.arctan(0,0)=0 -> zero constituents set to -np.pi
    """
    px = x[:,:,1]
    py = x[:,:,2]
    
    phis = np.arctan2(py,px)
    phis[phis < 0] += 2*np.pi
    phis[phis > 2*np.pi] -= 2*np.pi
    phis = phis - np.pi 
    return phis

def get_pT(x)->np.ndarray:
    px = x[:,:,1]
    py = x[:,:,2]
    return np.sqrt(px**2+py**2)

def get_pT_jet(x):
    px_sum = x[:,:,1].sum(axis=1)
    py_sum = x[:,:,2].sum(axis=1)
    return np.sqrt(px_sum**2+py_sum**2)

def create_plots(set:str, names:list[str], config:Settings):
    print("Creating Plots:")
    
    for name in names:
        set_path = f"{config.path_merged}/{set}/{name}"
        plot_path = f"{set_path}/plots"
        
        try: 
            x = np.load(f"{set_path}/x_data_prep.npy")
        except IOError:
            print("Set does not exist!")
            return
        try: 
            y = np.load(f"{set_path}/y_data_prep.npy")
        except IOError:
            print("Set does not exist!")
            return
    
        sig = x[np.where(y[:,0] == 1)]
        bkg = x[np.where(y[:,0] == 0)]
        
        label_list = ['E', 'p_x', 'p_y', 'p_z']
        X_label_list = ['Energy $E$ in [GeV]', 'Impuls $p_x$ in [GeV/c]', 'Impuls $p_y$ in [GeV/c]', 'Impuls $p_z$ in [GeV/c]']        
        legend = ['Signal', 'Hintergrund']
        histtype = "barstacked"
        
        
        for i in range(0,4):
            hist(x=[sig[:,:,i].ravel(), bkg[:,:,i].ravel()], labels=legend, Y_label="Number of events $N$", X_label=X_label_list[i], Y_scale="log", title=f"${label_list[i]}$ Signal", path=plot_path, fname=f"sign_{label_list[i]}", bins=100, histtype=histtype)
            #hist(x=[], Y_label="Number of events $N$", X_label=X_label_list[i], Y_scale="log", title=f"${label_list[i]}$ Background", path=plot_path, fname=f"bckg_{label_list[i]}", bins=100, histtype=histtype)
        
        hist(x=[get_pT(sig).ravel(), get_pT(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$p_T$ in [GeV/c]", Y_scale="log", title="$p_T$ Signal", path=plot_path, fname=f"sign_pT", bins=100, histtype=histtype)
        hist(x=[eta(sig).ravel(), eta(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$\eta$", Y_scale="log", title="$\eta$ Signal", path=plot_path, fname=f"sign_eta", bins=100, histtype=histtype)
        hist(x=[phi(sig).ravel(), phi(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$\phi$ in [Rad]", Y_scale="log", title="$\phi$ Signal", path=plot_path, fname=f"sign_phi", bins=100, histtype=histtype)
        hist(x=[get_pT_jet(sig).ravel(), get_pT_jet(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$p_T$ in [GeV/c]", Y_scale="log", title="$p_T$ Jet Signal", path=plot_path, fname=f"sign_pT_jet", bins=100, histtype=histtype)
        hist(x=[eta_jet(sig).ravel(), eta_jet(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$\eta$", Y_scale="log", title="$\eta$ Jet Signal", path=plot_path, fname=f"sign_eta_jet", bins=100, histtype=histtype)
        
        # hist(x=get_pT(bkg).ravel(), Y_label="Number of events $N$", X_label="$p_T$ in [GeV/c]", Y_scale="log", title="$p_T$ Background", path=plot_path, fname=f"bckg_pT", bins=100, histtype=histtype)
        # hist(x=eta(bkg).ravel(), Y_label="Number of events $N$", X_label="$\eta$", Y_scale="log", title="$\eta$ Background", path=plot_path, fname=f"bckg_eta", bins=100, histtype=histtype)
        # hist(x=phi(bkg).ravel(), Y_label="Number of events $N$", X_label="$\phi$ in [Rad]", Y_scale="log", title="$\phi$ Background", path=plot_path, fname=f"bckg_phi", bins=100, histtype=histtype)
        # hist(x=get_pT_jet(bkg).ravel(), Y_label="Number of events $N$", X_label="$p_T$ in [GeV/c]", Y_scale="log", title="$p_T$ Jet Background", path=plot_path, fname=f"bckg_pT_jet", bins=100, histtype=histtype)
        # hist(x=eta_jet(bkg).ravel(), Y_label="Number of events $N$", X_label="$\eta$", Y_scale="log", title="$\eta$ Jet Background", path=plot_path, fname=f"bckg_eta_jet", bins=100, histtype=histtype)