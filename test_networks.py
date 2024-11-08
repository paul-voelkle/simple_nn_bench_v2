from utilities import HyperParams, TrainStats, TestResults, config, load_dataset, load_model, clear, separator, confirm, store_results, init_device, val_pass
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np 
import torch
from torch.utils.data import DataLoader
from plot_utils import plot_2d, hist, scatter, plot_settings
import os

# PATH = "trained_models"
# RESULTS_PATH = "test_results"
TICKS = np.linspace(start=0, stop=1, num=11)

def get_prediction(dataloader:DataLoader, model, params:HyperParams,  n_monte=30, results:TestResults=None):
    
    
    x = dataloader.dataset.imgs.to(params.device)
    pred = []
    
    if model.name == "BayesConvNet2D":
        for i in range(n_monte):
            print(f"Computing Monte Carlo sample {i+1} / {n_monte}")
            with torch.no_grad():
                pred.append(model(x)[:,0].detach().unsqueeze(1))
                
        pred = torch.stack(pred)
        mean = torch.mean(pred, axis=0)
        std = torch.std(pred, axis=0)
    
    else:
        with torch.no_grad():
            pred.append(model(x)[:,0].detach().unsqueeze(1))
        
        pred = torch.stack(pred)
        mean = torch.mean(pred, axis=0)
        std = torch.zeros_like(mean)
    
    
    return mean, std

def eval_net(model, dataloader:DataLoader, loss_fn, results:TestResults, n_perform:int):
    print("Calculating loss on test set...")
    results.test_loss = val_pass(dataloader, model, loss_fn, test_mode=True)
    
    #test_loss = 0.0

    print("Tagging test set...")
    with torch.no_grad():
        test_pred = model(dataloader.dataset.imgs)
    
    # for i in range(n_perform):
    #     print(f"Computing Performance sample {i+1} / {n_perform}")
    #     results.startTimer()
    #     with torch.no_grad():
    #         model(dataloader.dataset.imgs)
    #     results.appendTime()
    
    # results.evalTime()
    
    print("Compairing predictions with labels...")
    test_corr = (torch.round(test_pred[:,0])==dataloader.dataset.labels[:,0]).sum().item()

    results.test_corr = test_corr*100/len(test_pred)
    
    separator()
    print(f"Performance evaluation of {model.name}:")
    print("MSE-loss on test dataset: {}".format(results.test_loss))
    print("Accuracy on test dataset: {:.1f} %".format(results.test_corr))
    separator()
    print()
    
    return test_pred

def closest_point(array, tpr_p=0.5):
    dist = ((array-tpr_p)**2)
    return np.argmin(dist)


def test_model(model:str="", dataset:str=""):
    
    result_path = f"{config.path_results}/{model}"

    test_set = f"{dataset}/test"
    
    results = TestResults()
    results.path_default = result_path

    clear()
    separator()
    print(f"Testing model: {model}")
    separator()
    
    if os.path.exists(result_path):
        if confirm(f"Path {result_path} already exists. Previous results might be overwritten. Edit path?"):
            print(f"Enter new path (relative to {config.path_results})")
            result_path = f"{config.path_results}/{input()}"

    print(f"Creating directory {result_path}")
    os.makedirs(result_path, exist_ok=True)

    #load saved model, training stats and hyperparam
    model_path = f"{config.path_trained}/{model}"
    params = HyperParams().load(model_path)
    stats = TrainStats().load(model_path)
    params.device = init_device()
    print(stats.name)
    model = load_model(stats.name, params, model_path)

    try:
        stats.dataset.append(test_set)
        STATS_DATASET = True
    except:
        STATS_DATASET = False
    
    #load test datat set
    test_dl = load_dataset(name=test_set, params=params)
    
    #evaluate models accuracy on test dataset
    test_pred = eval_net(model, test_dl, params.loss_fn, results=results, n_perform=20)

    #get mean, std pred
    mean_pred, std_pred = get_prediction(test_dl, model, params, n_monte=10, results=results)
    results.mean_pred, results.std_pred = mean_pred.cpu().reshape(-1).numpy(), std_pred.cpu().reshape(-1).numpy() 
    
    font_scale = plot_settings.font_scale
    histtype = "barstacked"


    #plot and save training and validation losses
    plot_2d(x=[stats.trn_losses, stats.val_losses], 
            linestyle=['solid','solid'],
            labels=["Trainingsverluste", 
                    "Validierungsverluste"], 
            X_label="Epoche", 
            Y_label="BCELoss", 
            #xticks=ticks,
            #yticks=ticks,
            path=result_path,
            fname='train_val_loss.png')
    

    #calculate tpr, fpr, auc_score and background rejection at 0.3
    results.fpr, results.tpr, th = roc_curve(test_dl.dataset.labels[:,0].long().cpu(), test_pred[:,0].cpu())
    results.auc_score = roc_auc_score(test_dl.dataset.labels[:,0].long().cpu(), test_pred[:,0].cpu())
    results.bck_rej = 1/results.fpr[closest_point(results.tpr, tpr_p=0.3)]
    
    results.save(results.path_default)
    store_results(params, stats, results, result_path)
    
    #plot and save roc curve
    plot_2d(x=[results.fpr, results.rnd_class],
            y=[results.tpr, results.rnd_class], 
            labels=["AUC = {:.2f}".format(results.auc_score), 'Rnd classifier'], 
            X_label='FPR', 
            Y_label='TPR', 
            #xticks = TICKS,
            #yticks = TICKS,
            path=result_path,
            fname='roc_curve.png',
            linestyle=['solid', '--'])
    
    #plot and save roc curve with inverse FPR
    plot_2d(x=[results.tpr, results.rnd_class],
            y=[1/results.fpr, 1/results.rnd_class], 
            labels=['AUC = {:.2f}\n 1/FPR(0.3) = {:.0f}'.format(results.auc_score, results.bck_rej), 'Rnd classifier'], 
            X_label='1/FPR', 
            Y_label='TPR',
            Y_scale='log',
            path=result_path,
            fname='roc_curve_inv_fpr.png', 
            linestyle=['solid', '--'])
        
    #plot and save pred hist
    hist(x=[results.mean_pred],
        labels=f"{stats.name}",
        X_label="Vorhersage",
        Y_label="$N/N_{tot}$",
        bins=50,
        histtype=histtype,
        path=result_path,
        fname='pred_hist.png',
        vline=0.5,
        vline_style="dashed")
    
    #plot and save std hist
    hist(x=[results.std_pred],
        labels=f"{stats.name}",
        X_label="$\sigma$",
        Y_label="$N/N_{tot}$",
        bins=50,
        histtype=histtype,
        path=result_path,
        fname='std_hist.png')  

    #scatter mean and std  
    scatter(x=results.mean_pred,
            y=results.std_pred,
            X_label="Vorhersage",
            Y_label="$\sigma_{\mathrm{pred}}$",
            labels=f"{stats.name}",
            path=result_path,
            fname='scatter.png')


