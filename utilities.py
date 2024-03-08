import os
from tabulate import tabulate
from print_data_utils import *
from torch.utils.data import DataLoader
from torch import cuda, nn
import torch
import models
import numpy as np
import pickle
import math
import csv

##menu navigation and printing utils
def confirm(text:str)->bool:
    while True:
        print(f"{text} [yes/no]")
        answer = input()
        if answer == "yes":
            return True
        elif answer == "no":
            return False

def input_handler(command:str=""):
    if command == "":
        return "",[]
    return command.split()[0], command.split()[1:]

def command_handler(command:str, allowed_commands:list[str], functions:list, args=list[str]):
    if len(allowed_commands) != len(functions):
        print("Internal error: allowed commands != funtions")
        return
    else:
        for i in range(len(functions)):
            if allowed_commands[i] == command:
                functions[i](args)
                return
    print(f"{command} is not a valid command")

def argument_handler(args:list[str], allowed_args:list[str], functions:list) -> bool:
    if len(allowed_args) != len(functions):
        print("Internal Error! length of allowed_args != length of functions")
        return False
    else:
        for i in range(len(functions)):
            if args[0] == allowed_args[i]:
                functions[i](args[1:])
                return True
        return False

def max_arg_error(args: list[str], max_args:int):
    if args.__len__() > max_args:
        print(f"Expected {max_args} or less arguments, but got {args.__len__()}")
        return True
    return False

def min_arg_error(args: list[str], min_args:int):
    if args.__len__() < min_args:
        print(f"Expected {min_args} or or more arguments, but got {args.__len__()}")
        return True
    return False

def invalid_args_error(args:list[str]):
    print(f"{args} is/are not valid argument(s)")


##class to track model training statistics
class TrainStats():
    
    stopTime = 0.0
    
    def __init__(self, 
                epoch:int = 0, 
                epochs:int = 0, 
                trn_loss:float = 0, 
                val_loss:float = 0, 
                trn_losses:list[float] = [],
                val_losses:list[float] = [],
                elapsed_time:float = 0, 
                trig:int = 0, 
                val_slope:float = -1, 
                batch_curr:int = 0, 
                batch_tot:int = 0,
                name:str = "",
                dataset:list[str] = [],
                device:str = "",
                early_stopping:bool = False):
        
        self.epoch = epoch
        self.epochs = epochs
        self.trn_loss = trn_loss
        self.trn_losses = trn_losses
        self.val_loss = val_loss
        self.val_losses = val_losses
        self.start_time = 0.0
        self.elapsed_time = elapsed_time
        self.trig = trig
        self.val_slope = val_slope
        self.batch_curr = batch_curr
        self.batch_tot = batch_tot
        self.name = name
        self.device = device
        self.early_stopping = early_stopping
        self.dataset = dataset
        self.startTimer()
    
    def update(self):
        clear()
        separator()
        print("Training model {}".format(self.name))
        print("Early stopping: {}".format(self.early_stopping))     
        print(f"Device: {self.device}")
        separator()
        print(f"Epoch {self.epoch} of {self.epochs}")
        separator()
        
        print("current batch loss: {} [{}/{}]".format(self.trn_loss, self.batch_curr, self.batch_tot))
        
        separator()
        print(f"Train Loss: {self.trn_loss}")
        print(f"Validation Loss: {self.val_loss}")
        print(f"Trigger count: {self.trig}")
        print(f"Slope of last 10 value losses: {self.val_slope}")
        self.getTimer()
        m, s = formatTime(self.elapsed_time)
        print(f"Elapsed Time: {m} min {s} s")
        separator()
        
    def startTimer(self):
        self.start_time = time.time()
    
    def save(self, path):
        
        os.makedirs(path, exist_ok=True)
        print(f"Saving Training Stats to {path}/hyper_parameter.pkl")
        with open(f"{path}/stats.pkl",'wb') as file:
            pickle.dump(self,file)
    

    def load(path):
        try:
            with open(f"{path}/stats.pkl", 'rb') as file:
                return pickle.load(file)
        except:
            print(f"{path}/stats.pkl not found!")
            raise

    def getTimer(self):
        self.elapsed_time = round(time.time() - self.start_time,3)
        
    def stopTimer(self):
        self.stopTime = time.time()
        
    def resumeTimer(self):
        self.start_time += time.time() - self.stopTime


##class for models training hyper parameters
class HyperParams():
    def __init__(self,
                 batch_size:int = 64,
                 epochs:int = 300,
                 loss_fn:nn.Module = nn.BCELoss(),
                 patience:int = 10,
                 val_slope_sample_length:int = 10,
                 max_val_loss_slope:float = -1e-06,
                 lr:float = 5e-4,
                 optimizer:torch.optim = None,
                 early_stopping:bool = True,
                 training_size:int = 0):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.patience = patience
        self.max_val_slope = max_val_loss_slope
        self.val_sample_length = val_slope_sample_length
        self.lr = lr
        self.optimizer = optimizer
        self.device = init_device() 
        self.early_stopping = early_stopping
        self.training_size = training_size

    def print_param(self):
        print(tabulate([["Optimizer","optimizer", self.optimizer, "Loss Function", "loss_fn", self.loss_fn],
                        ["Batch Size", "batch_size ", self.batch_size, "Epochs","epochs", self.epochs],
                        ["Learning Rate","lr", self.lr,"Early Stopping","early_stopping", self.early_stopping],
                        ["Early Stopping Patience","patience", self.patience, "Max. Slope of val Losses", "max_val_slope", self.max_val_slope],
                        ["Sample length for Slope", "val_sample_length", self.val_sample_length]]
                       , headers=["Description","Name","Value", "Description", "Name","Value"]))

    def save(self, path):
        
        os.makedirs(path, exist_ok=True)
        print(f"Saving Hyper Parameter to {path}/hyper_parameter.pkl")
        with open(f"{path}/hyper_parameter.pkl",'wb') as file:
            pickle.dump(self,file)
    

    def load(path):
        #try:
        with open(f"{path}/hyper_parameter.pkl", 'rb') as file:
            return pickle.load(file)
        #except:
        #    print(f"{path}/hyper_parameter.pkl not found!")
        #    raise
    
    def edit(self):
        if confirm("Edit Hyper Parmeters?"):
            print("Editing parameters. To show parameters:Show, To exit type:done, Usage: parameterName parameterValue")
            while True:
                command, arg = input_handler(input())
                if command == "done":
                    self.print_param()
                    break
                elif command == "show":
                    self.print_param()
                else:
                    try:
                        print(f"Setting {command} to {arg[0]}")
                        attr_type = type(getattr(self, command))
                        setattr(self, command, cast(arg[0],attr_type))
                    except:
                        print(f"{command} is not a valid attribute")


##functions for training and testing the models

def init_device():
    device = "cuda:0" if cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def val_pass( dataloader, model, loss_fn ):
    
    size = len( dataloader.dataset )
    num_batches = len( dataloader )
    nls = 0.0
    kls = 0.0
    vls = 0.0

    # we don't need gradients here since we only use the forward pass
    with torch.no_grad():
        for X, y in dataloader:
            if model.name == "BayesConvNet2D":
                pred = model( X )
                nl = loss_fn(pred, y)
                kl = model.KL()
                vl = nl.item() + kl.item()
                nls += nl
                kls += kl
                vls += vl
            else:
                pred = model( X )
                vls += loss_fn( pred, y ).item()

    nls /= num_batches
    kls /= num_batches
    vls /= num_batches
    return vls, nls, kls

def load_dataset(name:str="", path:str="data/preprocessed", trainSetBool:bool=False, params:HyperParams=HyperParams())->DataLoader:
    
    try:
        z = torch.Tensor(np.load(f"{path}/{name}/z_data.npy").reshape(-1, 1, 40,40).astype('float32')).to(params.device)
    except:
        raise
    
    try:
        y = torch.Tensor(np.load(f"{path}/{name}/y_data.npy")).to(params.device)
    except:
        print(f"{path}/{name}/y_data.npy does not exist! Return to main menu")
        raise

    if trainSetBool:
        params.training_size = np.load(f"{path}/{name}/y_data.npy").shape[0]

    dataset = models.dataset.dataset(z,y)    
    return DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)

def init_hyperparams():
    params = HyperParams()
    separator()
    print("Current Hyper Parameters:")
    separator()
    params.print_param()
    separator()
    params.edit()
            
    return params

def init_model(name:str, params:HyperParams):
    print(f"Initializing {name}")
    model_class = getattr(models, name)
    model = model_class.Model(training_size=params.training_size).to(params.device)
    print(model)
    return model

def load_model(name:str, params:HyperParams, filepath:str):
    print(f"Loading {name}")
    model_class = getattr(models, name)
    model = model_class.Model(training_size=params.training_size).to(params.device)
    model.eval()
    state_dict = torch.load(filepath + "/state.pth", map_location=params.device)
    model.load_state_dict(state_dict, strict=False)
    print(model)
    return model

def store_results(params:HyperParams, train_stats:TrainStats, test_loss, perc_corr, auc_score, path):
    print(f"Writing result to {path}/results.csv")
    with open(f"{path}/results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hyper Parameter'])
        writer.writerow(['Batch Size', 'Learning Rate', 'Device', 'Loss Function', 'Early Stopping', 'Patcience', 'Maximal Validation Loss Slope'])
        writer.writerow([params.batch_size, params.lr, params.device, params.loss_fn._get_name(), params.early_stopping, params.patience, params.max_val_slope])
        writer.writerow(['Training Statistics'])
        writer.writerow(['Model Name', 'Elapsed Time', 'Total Epochs', 'Training Dataset', 'Validation Dataset'])
        writer.writerow([train_stats.name, train_stats.elapsed_time, train_stats.epoch, train_stats.dataset[0], train_stats.dataset[1]])
        writer.writerow(['Testing Statistics'])
        writer.writerow(['Testing Dataset', 'Mean Test Loss', 'Correct Percentage', 'AUC Score'])
        writer.writerow([train_stats.dataset[2], test_loss, perc_corr, auc_score])


##further utilities
def cast(input, in_type:type):
    if in_type == int:
        return int(input)
    elif in_type == str:
        return str(input)
    elif in_type == float:
        return float(input)
    elif in_type == bool:
        return bool(input)
    else:
        return input

def split_array(split_size:int=10000, array:np.ndarray = None)->list:
    if array.any() == None:
        print("Error no Inputarrays")
        return

    num_of_splits = math.floor(len(array)/split_size)
    
    arrays =[]
    
    for i in range(0,num_of_splits):
        arrays.append(array.take(range(i*split_size,(i+1)*split_size),0))
    
    if len(array)%split_size != 0:
        arrays.append(array[(num_of_splits)*split_size:len(array),:,:])

    return arrays

def merge_arrays(arrays:list[np.ndarray])->np.ndarray:
    array = arrays[0]
    for i in range(1,len(arrays)):
        array = np.concatenate((array,arrays[i]), axis=0)

    return array
