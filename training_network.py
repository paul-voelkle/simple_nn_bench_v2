import os
import models
from plot_utils import plot_2d
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from utilities import TrainStats, HyperParams, config, val_pass, confirm, load_dataset, separator, init_model

def train_epoch(dataloader, model, loss_fn, optimizer, TrainStats:TrainStats):
    
    size = len( dataloader.dataset )
    TrainStats.batch_tot = size
    
    for batch, (X, y) in enumerate(dataloader):

        # pass data through network
        pred = model(X)
        
        # compute loss
        if model.name == "BayesConvNet2D":
            nl = loss_fn(pred, y)
            kl = model.KL()
            loss = nl + kl
        else:            
            loss = loss_fn(pred, y)

        # reset gradients in optimizer
        optimizer.zero_grad()
        
        # compute gradients
        loss.backward()
        
        # update weights with optimizer
        optimizer.step()
        
        # print the training loss every 100 updates
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            TrainStats.batch_curr = current
            TrainStats.trn_loss = loss
            TrainStats.update()
            #print(f"y pred: ({round(pred[0,0].item(),3)},{round(pred[0,1].item(),3)}), y actual: ({y[0,0].item()},{y[0,1].item()})")
            #print( f"current batch loss: {loss:>7f}  [{current:>5d}/{size:>5d}]" )

def getSlope(Y:np.ndarray, length:int=10):
    x = np.array(list(range(0, length))).reshape(-1,1)
    y = Y[-length:]
    model = LinearRegression()
    model.fit(x,y)
    model.score(x,y)
    
    return model.coef_

def train_loop(model, train_dl, val_dl, params:HyperParams, TrainStats:TrainStats):

    TrainStats.early_stopping = params.early_stopping
    TrainStats.name = model.name
    TrainStats.epochs = params.epochs

    val_corr_arr = []

    val_length_updated = False
    
    #epochs_of_early_stopping = []

    for t in range(params.epochs):
        TrainStats.epoch = t
        
        train_epoch( train_dl, model, params.loss_fn, params.optimizer, TrainStats)        
        
        TrainStats.trn_loss, nls, kls, = val_pass( train_dl, model, params.loss_fn )
        TrainStats.trn_losses.append( TrainStats.trn_loss )
        
        TrainStats.val_loss, nls, kls = val_pass( val_dl, model, params.loss_fn )
        TrainStats.val_losses.append( TrainStats.val_loss )        

        with torch.no_grad():    
            val_pred = model(val_dl.dataset.imgs)
        
        TrainStats.val_acc = (torch.round(val_pred[:,0])==val_dl.dataset.labels[:,0]).sum().item()/len(val_pred)
        TrainStats.val_accs.append(TrainStats.val_acc)
        
        #early stopping        
        if params.early_stopping and t > params.val_sample_length:
            TrainStats.val_slope = getSlope(np.array(TrainStats.val_losses), params.val_sample_length) 
        
        if TrainStats.val_slope >= params.max_val_slope:
            TrainStats.trig += 1
        else:
            TrainStats.trig = 0
        
        #plot losses and accuracy every 5 epochs
        if t%5 == 0:
            TrainStats.stopTimer()
            plot_2d(x=[TrainStats.trn_losses, TrainStats.val_losses], path='.', fname='current_training_losses.png', labels=["Trainingsverluste", "Validierungsverluste"], title=model.name, scale=1)
            plot_2d(x=[TrainStats.val_accs], path='.', fname='current_training_accuracy.png', labels=["Genauigkeit auf den val. Daten"], title=model.name, scale=1)
            TrainStats.resumeTimer()            
        
        
        #update learning rate every 30 epochs
        if params.lr_decr and t%params.lr_interv == 0:
            params.lr = params.lr*params.lr_decr_fact
            for g in params.optimizer.param_groups:
                g['lr'] = params.lr                 
        
        #update val sample length
        if int(TrainStats.trig/2) >= params.patience and not val_length_updated:
            params.val_sample_length = int(t/2)
            val_length_updated = True
        
        if TrainStats.trig >= params.patience:
            TrainStats.stopTimer()
            plot_2d(x=[TrainStats.trn_losses, TrainStats.val_losses], path='.', fname='last_training.png', labels=["training losses", "validation losses"], title=model.name,  scale=1)
            #epochs_of_early_stopping.append(t)
            if confirm("Early stopping triggered! Stop?"):
                print("Done!")
                return 
            else:
                params.edit()
                for g in params.optimizer.param_groups:
                    g['lr'] = params.lr     
                TrainStats.resumeTimer()
                TrainStats.trig = 0
    
    print("Done!")
    plot_2d(x=[TrainStats.trn_losses, TrainStats.val_losses], path='.', fname='last_training.png', labels=["Training losses", "Validation losses"], title=[model.name])
    return 

def save_model(model:models, TrainStats:TrainStats, params:HyperParams):
    
    PATH = f"{config.path_trained}/{model.name}"

    if confirm("Save Model?"):
        print(f"Standart path: {PATH}")
        
        if confirm("Edit path?"):
            print("Enter new path:")
            PATH = input()
        
        print(f"Saving model data to {PATH}")
        os.makedirs(PATH, exist_ok=True)

        torch.save(model.state_dict(), f"{PATH}/state.pth")
        TrainStats.save(path=PATH)
        params.save(path=PATH)
    else:
        print("Model not saved.")


def train_network(model_name:str, dataset:str):
    
    train_set = f"{dataset}/train"
    val_set = f"{dataset}/val"
    
    print("Loading last Hyperparameters:")
    try:
        params = HyperParams().load(".")
    except:
        print("No last Hyperparameter found. Using default settings")
        params = HyperParams()
    
    separator()
    print("Current Hyper Parameters:")
    separator()
    params.print_param()
    separator()
    params.edit()
    params.save(".")
    
    try:
        train_dl = load_dataset(name=train_set, params=params, trainSetBool=True)
    except:
        return

    try:
        val_dl = load_dataset(name=val_set, params=params)
    except:
        return
    
    model = init_model(name=model_name, params=params)
    
    stats = TrainStats()
    stats.device = params.device
    stats.dataset = [train_set, val_set]
    
    params.optimizer = torch.optim.Adam(params=model.parameters(), lr=params.lr)
    train_loop(model=model, params=params, train_dl=train_dl, val_dl=val_dl, TrainStats=stats)
    save_model(model=model, TrainStats=stats, params=params)
    return

