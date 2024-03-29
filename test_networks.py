from utilities import HyperParams, TrainStats,  Settings, load_dataset, load_model, clear, separator, confirm, store_results, init_device, val_pass
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np 
import torch
from torch.utils.data import DataLoader
from plot_utils import plot_2d, hist, scatter
import os

# PATH = "trained_models"
# RESULTS_PATH = "test_results"
TICKS = np.linspace(start=0, stop=1, num=11)

def get_prediction(dataloader:DataLoader, model, params:HyperParams,  n_monte=30):
    
    x = dataloader.dataset.imgs.to(params.device)
    pred = []
    for i in range(n_monte):
        print(f"Computing Monte Carlo sample {i+1} / {n_monte}")
        pred.append(model(x)[:,0].detach().unsqueeze(1))
        
    pred = torch.stack(pred)
    mean = torch.mean(pred, axis=0)
    std = torch.std(pred, axis=0)
    
    return mean, std

def eval_net(model, dataloader:DataLoader, loss_fn):
    test_loss = val_pass(dataloader, model, loss_fn)
    
    with torch.no_grad():
        test_pred = model(dataloader.dataset.imgs)
    
    test_corr = (torch.round(test_pred[:,0])==dataloader.dataset.labels[:,0]).sum().item()

    test_corr_prec = test_corr*100/len(test_pred)
    
    separator()
    print(f"Performance evaluation of {model.name}:")
    print("MSE-loss on test dataset: {}".format(test_loss))
    print("Accuracy on test dataset: {:.1f} %".format(test_corr_prec))
    separator()
    print()
    
    return test_pred, test_loss, test_corr_prec

def closest_point(array, tpr_p=0.3):
    dist = ((array-tpr_p)**2)
    return np.argmin(dist)


def test_model(path:str="", dataset:str="", config:Settings=None):
    
    if config == None:
        return
    
    result_path = f"{config.path_results}/{path}"

    test_set = f"{dataset}/test"
    
    clear()
    separator()
    print(f"Testing model: {path}")
    separator()
    
    if os.path.exists(result_path):
        if confirm(f"Path {result_path} already exists. Previous results might be overwritten. Edit path?"):
            print(f"Enter new path (relative to {config.path_results})")
            result_path = f"{config.path_results}/{input()}"

    print(f"Creating directory {result_path}")
    os.makedirs(result_path, exist_ok=True)

    #load saved model, training stats and hyperparam
    model_path = f"{config.path_trained}/{path}"
    params = HyperParams.load(model_path)
    stats = TrainStats.load(model_path)
    params.device = init_device()
    print(stats.name)
    model = load_model(stats.name, params, model_path)

    try:
        stats.dataset.append(test_set)
        STATS_DATASET = True
    except:
        STATS_DATASET = False
    
    #show hyperparamter edit promt
    separator()
    print("Current Hyper Parameters:")
    separator()
    params.print_param()
    separator()
    params.edit()

    #load test datat set
    test_dl = load_dataset(name=test_set, params=params, config=config)

    #evaluate models accuracy on test dataset
    test_pred, test_loss, test_corr_perc = eval_net(model, test_dl, params.loss_fn)

    #get mean, std pred
    mean_pred, std_pred = get_prediction(test_dl, model, params, n_monte=10)
    mean_pred, std_pred = mean_pred.cpu().reshape(-1).numpy(), std_pred.cpu().reshape(-1).numpy() 
  
    #plot and save training and validation losses
    plot_2d(x=[stats.trn_losses, stats.val_losses], 
            Y_scale='log', 
            linestyle=['solid','solid'],
            labels=["training losses", 
                    "validation losses"], 
            X_label="epoch", 
            Y_label="Binary CE", 
            #xticks=ticks,
            #yticks=ticks,
            path=result_path,
            fname='train_val_loss.png', 
            title=f'{stats.name}: training and validation losses')
        
    #plot and save roc and auc score
    fpr, tpr, th = roc_curve(test_dl.dataset.labels[:,0].long().cpu(), test_pred[:,0].cpu())
    auc_score = roc_auc_score(test_dl.dataset.labels[:,0].long().cpu(), test_pred[:,0].cpu())
    rnd_class = np.linspace(0, 1, 100)
    
        
    #plot and save roc curve
    plot_2d(x=[fpr, rnd_class],
            y=[tpr, rnd_class], 
            title=f'{stats.name}: ROC-Curve', 
            labels=["AUC = {:.2f}".format(auc_score), 'Rnd classifier'], 
            X_label='$\epsilon_{bkg}$ - FPR', 
            Y_label='$\epsilon_{s}$ - TPR', 
            xticks = TICKS,
            yticks = TICKS,
            path=result_path,
            fname='roc_curve.png',
            linestyle=['solid', '--'])
    
    #plot and save roc curve with inverse FPR
    plot_2d(x=[tpr, rnd_class],
            y=[1/fpr, 1/rnd_class], 
            title=f'{stats.name}: ROC-Curve - Inverse FPR', 
            labels=['AUC = {:.2f}\n $1/\epsilon_{{bkg}}$(0.3) = {:.0f}'.format(auc_score, 1/fpr[closest_point(tpr, tpr_p=0.3)]), 'Rnd classifier'], 
            X_label='$\epsilon_{bkg}$ - FPR', 
            Y_label='$\epsilon_{s}$ - TPR',
            Y_scale='log',
            path=result_path,
            fname='roc_curve_inv_fpr.png', 
            linestyle=['solid', '--'])
        
    #plot and save pred hist
    hist(x=mean_pred,
        title=f"{stats.name}: Histogramm of predicted values",
        X_label="Prediction",
        Y_label="Normailized",
        bins=50,
        histtype="step",
        path=result_path,
        fname='pred_hist.png')
    
    #plot and save std hist
    hist(x=std_pred,
        title="{stats.name}: Histogramm of standard deviation",
        X_label="$\sigma_{\mathrm{pred}}$",
        Y_label="Normailized",
        bins=50,
        histtype="step",
        path=result_path,
        fname='std_hist.png',
        density=True)  

    #scatter mean and std  
    scatter(x=mean_pred,
            y=std_pred,
            X_label="$\mu$",
            Y_label="$\sigma_{\mathrm{pred}}$",
            title=f"{stats.name}: Scatter Plot",
            path=result_path,
            fname='scatter.png')
    
    store_results(params, stats, test_loss, test_corr_perc, auc_score, result_path)

