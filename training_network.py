import os
import models
from print_data_utils import *
import time
from plot_utils import plot_2d
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from utilities import *


def train_epoch( dataloader, model, loss_fn, optimizer, TrainStats:TrainStats):
    
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
        if batch % 100 == 0:
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


    #epochs_of_early_stopping = []

    for t in range(params.epochs):
        TrainStats.epoch = t
        
        train_epoch( train_dl, model, params.loss_fn, params.optimizer, TrainStats)        
        TrainStats.trn_loss, nls, kls, = val_pass( train_dl, model, params.loss_fn )
        TrainStats.trn_losses.append( TrainStats.trn_loss )
        
        TrainStats.val_loss, nls, kls = val_pass( val_dl, model, params.loss_fn )
        TrainStats.val_losses.append( TrainStats.val_loss )        

        #early stopping        
        if params.early_stopping and t > params.val_sample_length:
            TrainStats.val_slope = getSlope(np.array(TrainStats.val_losses), params.val_sample_length) 
        if TrainStats.val_slope >= params.max_val_slope:
            TrainStats.trig += 1
        else:
            TrainStats.trig = 0
        
        if TrainStats.trig >= params.patience:
            TrainStats.stopTimer()
            plot_2d(x=[TrainStats.trn_losses, TrainStats.val_losses], path='.', fname='last_training.png', labels=["Training losses", "Validation losses"], title=[model.name])
            #epochs_of_early_stopping.append(t)
            if confirm("Early stopping triggered! Stop?"):
                print("Done!")
                return 
            else:
                TrainStats.resumeTimer()
                TrainStats.trig = 0
    
    print("Done!")
    plot_2d(x=[TrainStats.trn_losses, TrainStats.val_losses], path='.', fname='last_training.png', labels=["Training losses", "Validation losses"], title=[model.name])
    return 

def save_model(model:models, TrainStats:TrainStats, params:HyperParams):
    PATH = f"trained_models/{model.name}"

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


def train_network(model_name:str, train_set:str, val_set:str):
    param = init_hyperparams()
    stats = TrainStats(device=param.device, dataset=[train_set, val_set])
    try:
        train_dl = load_dataset(name=train_set, params=param, trainSetBool=True)
    except:
        return

    try:
        val_dl = load_dataset(name=val_set, params=param)
    except:
        return
    
    model = init_model(name=model_name, params=param)
    param.optimizer = torch.optim.Adam(params=model.parameters(), lr=param.lr)
    train_loop(model=model, params=param, train_dl=train_dl, val_dl=val_dl, TrainStats=stats)
    save_model(model=model, TrainStats=stats, params=param)
    return

