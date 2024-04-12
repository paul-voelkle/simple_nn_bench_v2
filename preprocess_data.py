import sys
import numpy as np
from plot_utils import heatmap
import os
from utilities import merge_arrays, split_array, confirm, Settings

MAX_DATA_LENGTH = 50000

NORM = True
ROTATE = False
FLIP = False

# Some initial settings
__n_warning__ = 0.7
n_shift_phi, n_shift_eta = 0, 0

# Grid settings
xpixels = np.arange(-2.6, 2.6, 0.029)
ypixels = np.arange(-np.pi, np.pi, 0.035)

# Calculate the pseudorapidity of pixel entries
def eta (pT, pz):
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

# Calculate the azimuthal angle of pixel entries
def phi (px, py):
    """
    phis are returned in rad., np.arctan(0,0)=0 -> zero constituents set to -np.pi
    """
    phis = np.arctan2(py,px)
    phis[phis < 0] += 2*np.pi
    phis[phis > 2*np.pi] -= 2*np.pi
    phis = phis - np.pi 
    return phis

# function to calculate masses
def mass (E,px,py,pz):
    mass = np.sqrt(np.maximum(0.,E**2-px**2-py**2-pz**2))
    return mass

# function to return the image momenta for centroid and principal axis
def img_mom (x, y, weights, x_power, y_power):
    return ((x**x_power)*(y**y_power)*weights).sum()

# returns the jet image
def orig_image (etas, phis, es):
    """
    Gives the value on grid with minimal distance,
    eg. for xpixel = (0,1,2,3,..) eta=1.3 -> xpixel=1, eta=1.6 ->xpixel=2
    """
    z = np.zeros((etas.shape[0],len(xpixels),len(ypixels)))
    in_grid = ~((etas < xpixels[0]) | (etas > xpixels[-1]) | (phis < ypixels[0]) | (phis > ypixels[-1]))
    xcoords = np.argmin(np.abs(etas[:,None,:] - xpixels[None,:,None]),axis=1)
    ycoords = np.argmin(np.abs(phis[:,None,:] - ypixels[None,:,None]),axis=1)
    ncoords = np.repeat(np.arange(etas.shape[0])[:,None],etas.shape[1],axis=1)
    z[ncoords[in_grid],ycoords[in_grid],xcoords[in_grid]] = es[in_grid]
    return z

# preprocess the jet
def preprocessing( x ,y, weights, rotate=True, flip=True ):
    """
    (x,y) are the coordinates and weights the corresponding values, shifts
    centroid to origin, rotates image, so that principal axis is vertical,
    flips image, so that most weights lay in (x<0, y>0)-plane.
    Method for calculating principal axis (similar to tensor of inertia):
    https://en.wikipedia.org/wiki/Image_moment
    here: y=phi, phi has modulo 2*np.pi but it's not been taken care of hear,
    so possible issues with calculating the centroid
    -> pre-shifting of events outside of this function solves the problem
    for image-data with Delta_phi < 2*np.pi
    """

    # Shift
    x_centroid = img_mom(x, y, weights, 1, 0) / weights.sum()
    y_centroid = img_mom(x, y, weights, 0, 1)/ weights.sum()
    x = x - x_centroid
    y = y - y_centroid

    # Check if shifting worked, there can be problems with modulo variables like phi (y)
    # x and y are sorted after highest weight, 0-comp. gives hottest event
    # for Jet-like Images Centroid should be close to hottest constituen (pT-sorted arrays)  
    global n_shift_phi
    global n_shift_eta
    if np.abs(x[0]) > __n_warning__:
        n_shift_eta += 1  
    if np.abs(y[0]) > __n_warning__:
        n_shift_phi += 1       

    if rotate:
        #Covariant matrix, eigenvectors corr. to principal axis
        u11 = img_mom(x, y, weights, 1, 1) / weights.sum()
        u20 = img_mom(x, y, weights, 2, 0) / weights.sum()
        u02 = img_mom(x, y, weights, 0, 2) / weights.sum()
        cov = np.array([[u20, u11], [u11, u02]])

        # Eigenvalues and eigenvectors of covariant matrix
        evals, evecs = np.linalg.eig(cov)

        # Sorts the eigenvalues, v1, [::-1] turns array around, 
        sort_indices = np.argsort(evals)[::-1]
        e_1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        e_2 = evecs[:, sort_indices[1]]

        # Theta to x_asix, arctan2 gives correct angle
        theta = np.arctan2(e_1[0], e_1[1])
  
        # Rotation, so that princple axis is vertical
        # anti-clockwise rotation matrix
        rotation = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation * np.stack([x,y])
        x_rot, y_rot = transformed_mat.A
    else: 
        x_rot, y_rot = x, y
  
    # Flipping
    n_flips = 0
    if flip:
        if weights[x_rot<0.].sum() < weights[x_rot>0.].sum():
            x_rot = -x_rot
            n_flips  += 1
        if weights[y_rot<0.].sum() > weights[y_rot>0.].sum():
            y_rot = -y_rot
            n_flips += 1
            
    return x_rot, y_rot


def pT_cut(x:np.ndarray, y:np.ndarray, pT_min:float, pT_max:float):
    print("Cropping jets")
    pxs   = x[:,:,1]
    pys   = x[:,:,2]
    pT    = np.sqrt(pxs**2+pys**2).sum(axis=1)
    return np.delete(x, np.where((pT <= pT_min) | (pT >= pT_max)), axis=0), np.delete(y, np.where((pT <= pT_min) | (pT >= pT_max)),axis=0)

#function to remove entries with zero pT
def remove_zero_pT(x:np.ndarray,y:np.ndarray)->np.ndarray:
    print("Removing entries with zero pT")
    return pT_cut(x,y,0.0,sys.float_info.max)
            
# function to convert the jet to an image
def constit_to_img( jets:np.ndarray, n_constit:int, norm:bool, rotate:bool, flip:bool ):
    
    print( "Crop constituents" )
    jets = jets[:,0:n_constit,:]
    
    print( "Calculating pT" )
    E     = jets[:,:,0]
    pxs   = jets[:,:,1]
    pys   = jets[:,:,2]
    pzs   = jets[:,:,3]
    pT    = np.sqrt(pxs**2+pys**2)
    

    print( "Calculating eta" )
    etas  = eta(pT,pzs)
    
    print( "Calculating phi" )
    phis  = phi(pxs,pys)
    
    print( "Calculating the mass" )
    E_tot = E.sum(axis=1)
    px_tot = pxs.sum(axis=1)
    py_tot = pys.sum(axis=1)
    pz_tot = pzs.sum(axis=1)
    j_mass = mass(E_tot, px_tot, py_tot, pz_tot)
    
    print( "Pre-shifting the phis" )
    phis = (phis.T - phis[:,0]).T
    phis[phis < -np.pi] += 2*np.pi
    phis[phis > np.pi] -= 2*np.pi
    
    print( "Using pT as weight" )
    weights = pT
    
    print( "Preprocessing" )
    
    for i in range( np.shape(etas)[0] ):
        etas[i,:], phis[i,:] = preprocessing( etas[i,:], phis[i,:], weights[i,:], rotate, flip )
    
    print( "Creating images" )
    z_ori = orig_image(etas, phis, weights)
    
    #return z_ori
        
    print( "Cropping and normalising" )
    n_crop = 40
    z_new = np.zeros( (z_ori.shape[0],n_crop, n_crop) )
    for i in range(z_ori.shape[0]):
        Npix = z_ori[i,:,:].shape
        z_new[i,:,:] = z_ori[i, int(Npix[0]/2-n_crop/2) : int(Npix[0]/2+n_crop/2), int(Npix[1]/2-n_crop/2) : int(Npix[1]/2+n_crop/2) ]
        if norm:
            z_sum = z_new[i,:,:].sum()
            if z_sum != 0.:
                z_new[i,:,:] = z_new[i,:,:]/z_sum
    
    print( "Reshaping" )
    z_out = z_new.reshape( (z_new.shape[0],-1) )
    
    return z_out

def getPt(event:np.ndarray):
    px = event[:,1]
    py = event[:,2]
    return np.sqrt(px**2 + py**2)

def sortEventsPt(arr:np.ndarray):
    if len(arr) <= 1:
        return arr
    else:
        pivot = getPt(arr)[int(len(arr)/2)]
        left = arr[np.where( getPt(arr) > pivot )]
        middle = arr[np.where( getPt(arr) == pivot )]
        right = arr[np.where( getPt(arr) < pivot )]
        return np.concatenate((sortEventsPt(left),middle,sortEventsPt(right)),axis=0)

def sortPT(arr:np.ndarray):
    print("Sort event constituents")
    for i in range(len(arr)):
        arr[i] = sortEventsPt(arr[i])

def preprocess_data(src_folder:str="", files:list[str]=[], config:Settings=None):
    
    if config == None:
        return
    
    if len(files) == 0:
        print("Error: No filenames provided!")
        return
    
    if confirm("Set pT Cuts?"):
        print("Usage: ptMin[GeV] ptMax[GeV]")
        ptMin, ptMax = input().split()
        pTCut = True
    else:
        pTCut = False
    
    for name in files:
        file_path = f"{config.path_merged}/{src_folder}/{name}"
        x_filepath = f"{file_path}/x_data.npy"
        y_filepath = f"{file_path}/y_data.npy"
        y_filepath_prep = f"{file_path}/y_data_prep.npy"
        z_filepath = f"{file_path}/z_data.npy"
        
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        x = np.load(x_filepath)
        y = np.load(y_filepath)
        x, y = remove_zero_pT(x, y)
                
        if pTCut:
            x, y = pT_cut(x, y, float(ptMin), float(ptMax))
        
        if confirm(f"Limit size of {file_path}"):
            print("Enter size:")
            size = int(input())
            if size < len(x):
                print(f"Limiting size to {size}")
                x = x[0:size,:,:]
                y = y[0:size,:]        
        
        #split x array for preprocessing:
        if len(x) > MAX_DATA_LENGTH:
            xs = split_array(MAX_DATA_LENGTH, x)
            zs = []
            for x in xs:
                sortPT(x)
                zs.append(constit_to_img(x, 50, NORM, ROTATE, FLIP).astype('float32'))

            z = merge_arrays(zs)
        else:
            z = constit_to_img(x, 50, NORM, ROTATE, FLIP).astype('float32')
        
        sig = z[np.where( y[:,0] == 1)]
        bkg = z[np.where( y[:,0] == 0)]

        print(n_shift_eta, n_shift_phi)
        
        print(f"Done! Saving file to {file_path}")
        heatmap(sig.mean(0).reshape((40,40)),X_label="$\eta$", Y_label="$\phi$", title=f"Signal with {len(sig)} Jets", path=file_path, fname="Signal")
        heatmap(bkg.mean(0).reshape((40,40)),X_label="$\eta$", Y_label="$\phi$", title=f"Background with {len(bkg)} Jets", path=file_path, fname="Background")
        np.save(z_filepath,z)
        np.save(y_filepath_prep,y)

def merge_data(src:str, out:str, shuffle:bool, config:Settings):
    
    if config == None:
        return    
    
    print(f"Merging data {config.path_notmerged}/{src}")
    
    x_data_1 = np.load(f"{config.path_notmerged}/{src}/signal/x_data.npy")
    y_data_1 = np.load(f"{config.path_notmerged}/{src}/signal/y_data.npy")
    
    x_data_2 = np.load(f"{config.path_notmerged}/{src}/background/x_data.npy")
    y_data_2 = np.load(f"{config.path_notmerged}/{src}/background/y_data.npy")
    
    x_data = np.concatenate((x_data_1, x_data_2), axis=0)
    y_data = np.concatenate((y_data_1, y_data_2), axis=0)
    
    if shuffle:
        seed = np.random.randint(0, 100000) 
        print(f"Shuffle merged array with seed {seed}")
        np.random.seed(seed)  
        np.random.shuffle(x_data)
        np.random.seed(seed)          
        np.random.shuffle(y_data)        
    
    os.makedirs(f"{config.path_merged}/{out}", exist_ok=True)
    
    print(f"Saving to {config.path_merged}/{out}/")
    
    np.save(f"{config.path_merged}/{out}/x_data.npy", x_data)
    np.save(f"{config.path_merged}/{out}/y_data.npy", y_data)