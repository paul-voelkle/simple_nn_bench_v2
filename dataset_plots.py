import os
from plot_utils import hist, heatmap
from utilities import config
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

def create_plots(set:str, names:list[str]):
    print("Creating Plots:")
    
    for name in names:
        set_path = f"{config.path_merged}/{set}/{name}"
        plot_path = f"{set_path}/plots"
        
        try: 
            x = np.load(f"{set_path}/x_data_prep.npy")
        except IOError:
            print(f"Set {set_path} does not exist!")
            return
        try: 
            y = np.load(f"{set_path}/y_data_prep.npy")
        except IOError:
            print(f"Set {set_path} does not exist!")
            return
        try: 
            z = np.load(f"{set_path}/z_data.npy")
        except IOError:
            print(f"Set {set_path} does not exist!")
            return
        
    
        sig = x[np.where(y[:,0] == 1)]
        bkg = x[np.where(y[:,0] == 0)]
        
        sig_z = z[np.where(y[:,0] == 1)]
        bkg_z = z[np.where(y[:,0] == 0)]
        
        label_list = ['E', 'p_x', 'p_y', 'p_z']
        X_label_list = ['$E$ [GeV]', '$p_x$ [GeV/c]', '$p_y$ [GeV/c]', '$p_z$ [GeV/c]']        
        legend = ['Signal', 'Hintergrund']
        histtype = "barstacked"
        
        rand_sig = np.random.randint(0, len(sig_z)) 
        rand_bkg =  np.random.randint(0, len(bkg_z)) 
        
        for i in range(0,4):
            hist(x=[sig[:,:,i].ravel(), bkg[:,:,i].ravel()], labels=legend, Y_label="Number of constituents $N$", X_label=X_label_list[i], Y_scale="log", path=plot_path, fname=f"sign_{label_list[i]}", bins=100, histtype=histtype)
        
        hist(x=[get_pT(sig).ravel(), get_pT(bkg).ravel()], labels=legend, Y_label="Number of constituents $N$", X_label="$p_T$ [GeV/c]", Y_scale="log", path=plot_path, fname=f"sign_pT", bins=100, histtype=histtype)
        hist(x=[eta(sig).ravel(), eta(bkg).ravel()], labels=legend, Y_label="Number of constituents $N$", X_label="$\eta$", Y_scale="log", path=plot_path, fname=f"sign_eta", bins=100, histtype=histtype)
        hist(x=[phi(sig).ravel(), phi(bkg).ravel()], labels=legend, Y_label="Number of constituents $N$", X_label="$\phi$ [Rad]", Y_scale="log", path=plot_path, fname=f"sign_phi", bins=100, histtype=histtype)
        hist(x=[get_pT_jet(sig).ravel(), get_pT_jet(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$p_{T,J} }$ [GeV/c]", Y_scale="log", path=plot_path, fname=f"sign_pT_jet", bins=100, histtype=histtype)
        hist(x=[eta_jet(sig).ravel(), eta_jet(bkg).ravel()], labels=legend, Y_label="Number of events $N$", X_label="$\eta_J }$", Y_scale="log", path=plot_path, fname=f"sign_eta_jet", bins=100, histtype=histtype)
        
        # num_xticks = 10
        # x_start = -2.6
        # x_stop = 2.6
        # x_step = abs(x_start-x_stop)/num_xticks

        xpixels = np.array(["-2.6","","","","0","","","","2.6"])
        ypixels = np.array(["-$\pi$","","","","0","","","","$\pi$"])

        heatmap(sig_z.mean(0).reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", title=f"Gemitteltes Signal mit {len(sig_z)} Jets", path=plot_path, fname="signal_mean")
        heatmap(bkg_z.mean(0).reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", title=f"Gemittelter Hintergrund mit {len(bkg_z)} Jets", path=plot_path, fname="background_mean")
        
        rand_sig = np.random.randint(0, len(sig_z)) 
        rand_bkg =  np.random.randint(0, len(bkg_z))        
        heatmap(sig_z[rand_sig].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="signal_single_1")
        heatmap(bkg_z[rand_bkg].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="background_single_1")
        rand_sig = np.random.randint(0, len(sig_z)) 
        rand_bkg =  np.random.randint(0, len(bkg_z))
        heatmap(sig_z[rand_sig].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="signal_single_2")
        heatmap(bkg_z[rand_bkg].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="background_single_2")
        rand_sig = np.random.randint(0, len(sig_z)) 
        rand_bkg =  np.random.randint(0, len(bkg_z))        
        heatmap(sig_z[rand_sig].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="signal_single_3")
        heatmap(bkg_z[rand_bkg].reshape((40,40)), xticks=xpixels, yticks=ypixels, X_label="$\Delta \eta$", Y_label="$\Delta \phi$", path=plot_path, fname="background_single_3")
