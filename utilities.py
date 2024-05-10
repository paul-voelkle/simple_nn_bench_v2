import os
from typing import TypeVar
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
import io

T = TypeVar('T', bound='DataIO')

##menu navigation and printing utils
def confirm(text:str, default:bool=False)->bool:
    while True:
        if default: print(f"{text} [yes/no] (default: yes)") 
        else: print(f"{text} [yes/no] (default: no)")
        answer = input()
        if answer == "yes":
            return True
        elif answer == "no":
            return False
        elif answer == "":
            return default

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

def argument_handler(args:list[str], allowed_args:list[str], functions:list)->bool:
    if len(allowed_args) != len(functions):
        print("Internal Error! length of allowed_args != length of functions")
        return False
    else:
        for i in range(len(functions)):
            if args[0] == allowed_args[i]:
                functions[i](args[1:])
                return True
        return False

def max_arg_error(args: list[str], max_args:int)->bool:
    if args.__len__() > max_args:
        print(f"Expected {max_args} or less arguments, but got {args.__len__()}")
        return True
    return False

def min_arg_error(args: list[str], min_args:int)->bool:
    if args.__len__() < min_args:
        print(f"Expected {min_args} or or more arguments, but got {args.__len__()}")
        return True
    return False

def invalid_args_error(args:list[str]):
    print(f"{args} is/are not valid argument(s)")



class DataIO():    
    def __init__(self, filename:str, path_default:str, name:str, ver:str):
        self.name:str = name
        self.filename = filename
        self.path_default = path_default
        self.__noedit__ = ["__noedit__", "name", "filename", "path_default", "ver"]
        self.ver = ver
        
    def save(self, path:str):
        os.makedirs(path, exist_ok=True)
        print(f"Saving to {path}/{self.filename}")
        with open(f"{path}/{self.filename}",'wb') as file:
            pickle.dump(self,file)

    def load(self, path:str="")->T:     
        
        if path == "":
            path = self.path_default
        
        print(f"Loading {path}/{self.filename}")   
        try:
            with open(f"{path}/{self.filename}", 'rb') as file:
                if cuda.is_available():
                    return self.updater(pickle.load(file))
                else:
                    return self.updater(CPU_Unpickler(file).load())
        except IOError:
            print(f"{path}/{self.filename} not found!")
            return self
    
    def updater(self, file):
        try:
            ver = file.ver
        except:
            ver = "None"

        if self.ver != ver:
            print(f"Loaded file is an older version! Updating file...")
            old_params = list(file.__dict__.keys())
            
            for param in old_params:
                if param != "ver" or "__noedit__":
                    #print(f"Setting {param} = {getattr(file, param)}")
                    setattr(self,param, getattr(file, param)) 
            
            return self
        else:
            return file

    def print_param(self):
        row_length = 2
        params = list(self.__dict__.keys())
        headers = ["Name","Type","Value"]
        headers = headers*row_length
        table = []
        row = []
        
        for param in params:
            if len(row)%len(headers) == 0 and len(row) != 0:
                table.append(row)
                row = []
            
            if param not in self.__noedit__:
                row.append(param)
                row.append(type(getattr(self, param)))
                row.append(getattr(self, param))        
        
        if len(row) != 0:
            table.append(row)       
        
        print(tabulate(table, headers=headers))

    def edit(self):
        if confirm(f"Edit {self.name}?", True):
            print("Editing parameters. To show parameters:Show, To exit type:done, Usage: parameterName parameterValue")
            while True:
                command, arg = input_handler(input())
                if command == "done":
                    self.save(self.path_default)
                    break
                elif command == "show":
                    self.print_param()
                elif command not in self.__noedit__:
                    try:
                        print(f"Setting {command} to {arg[0]}")
                        attr_type = type(getattr(self, command))
                        setattr(self, command, cast(arg[0],attr_type))
                    except:
                        print(f"{command} is not a valid attribute")
                else: 
                    print(f"{command} is not a valid attribute")

##program settings
class Settings(DataIO):
    
    def __init__(self):
        super().__init__(filename="config.pkl", path_default=".", name="Program Settings", ver="0.0.1")        
        #paths for models and test results
        self.path_trained = "trained_models"
        self.path_results = "test_results"
        self.path_models = "models"
        
        #paths for datasets
        self.path_data = "data"
        self.path_notmerged = f"{self.path_data}/not_merged"
        self.path_merged = f"{self.path_data}/merged"

        #preprocessing configuration
        self.prep_norm:bool = True
        self.prep_flip:bool = True
        self.prep_rot:bool = True
        self.prep_max_data:int = 50000
    
    def load(self)->'Settings':
        return super().load(self.path_default)

    # def load(self):
    #     try:
    #         with open("config.pkl", 'rb') as file:
    #             print("Loading config.pkl")
    #             return pickle.load(file)
    #     except IOError:
    #         print("No config file found. Initialize factory settings")
    #         self.load_factory()
    #         return

    def create_dirs(self):
        for atr in self.__dict__.keys():
            if atr.startswith("path_"):
                os.makedirs(self.__getattribute__(atr), exist_ok=True)
    
    # def load_factory(self):
    #     self.__init__()
    #     self.save()
        
## cpu unpickler
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

##class to track model training statistics
class TrainStats(DataIO):
    
    stopTime = 0.0    
    
    def __init__(self):
        super().__init__(filename="stats.pkl", path_default=".", name="Training Stats", ver="0.0.1")
        self.epoch:int = 0
        self.epochs:int = 0
        self.trn_loss:float = 0.0
        self.trn_losses:list[float] = []
        self.val_loss:float = 0.0
        self.val_losses:list[float] = []
        self.val_acc:float = 0.0
        self.val_accs:list[float] = []
        self.start_time:float = 0.0
        self.elapsed_time:float = 0.0
        self.trig:int = 0
        self.val_slope:float = -1.0
        self.batch_curr:int = 0
        self.batch_tot:int = 0
        self.name:str = ""
        self.device:str = ""
        self.early_stopping:bool = True
        self.dataset:list[str] = []
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
        if self.epoch != 0:
            print(f"Last train Loss: {self.trn_losses[-1]}")
            print(f"Last validation Loss: {self.val_losses[-1]}")
            print(f"Last validation Accuracy: {np.round(self.val_accs[-1],4)*100} %")
        print(f"Trigger count: {self.trig}")
        print(f"Slope of last value losses: {self.val_slope}")
        self.getTimer()
        m, s = formatTime(self.elapsed_time)
        print(f"Elapsed Time: {m} min {s} s")
        separator()
        
    def startTimer(self):
        self.start_time = time.time()
    
    def getTimer(self):
        self.elapsed_time = round(time.time() - self.start_time,3)
        
    def stopTimer(self):
        self.stopTime = time.time()
        
    def resumeTimer(self):
        self.start_time += time.time() - self.stopTime


##class for model training hyper parameters
class HyperParams(DataIO):
    
    def __init__(self):
        super().__init__(filename="hyper_parameter.pkl", path_default=".", name="Hyper Parameter", ver="0.0.3")
        self.batch_size:int = 64
        self.epochs:int = 300
        self.loss_fn:nn.Module = nn.BCELoss()
        self.patience:int = 10
        self.max_val_slope:float = -1e-06
        self.val_sample_length:int = 10
        self.lr:float = 5e-5
        self.lr_decr:bool = False
        self.lr_interv:int = 30
        self.lr_decr_fact:float = 0.5
        self.optimizer:torch.optim = None
        self.early_stopping:bool = True
        self.training_size:int = 0
        self.device = "cuda:0" if cuda.is_available() else "cpu"
        self.__noedit__ = self.__noedit__ + ["optimizer"]


class TestResults(DataIO):
    
    def __init__(self):
        super().__init__(filename="results.pkl", path_default=".", name="test results", ver="0.0.1")
        self.fpr = []
        self.tpr = []
        self.auc_score = []
        self.rnd_res = 400
        self.rnd_class = np.linspace(0,1,self.rnd_res)



##functions for training and testing the models
def init_device():
    device = "cuda:0" if cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def val_pass( dataloader, model, loss_fn, test_mode:bool = False):
    
    size = len( dataloader.dataset )
    num_batches = len( dataloader )
    nls = 0.0
    kls = 0.0
    vls = 0.0

    # we don't need gradients here since we only use the forward pass
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
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
            if test_mode and batch % 50 == 0:
                print(f"Current batch: {batch}")
    nls /= num_batches
    kls /= num_batches
    vls /= num_batches
    return vls, nls, kls

def load_dataset(name:str="", trainSetBool:bool=False, params:HyperParams=HyperParams())->DataLoader:
        
    path = config.path_merged
    
    try:
        z = torch.Tensor(np.load(f"{path}/{name}/z_data.npy").reshape(-1, 1, 40,40).astype('float32')).to(params.device)
    except:
        print(f"{path}/{name}/z_data.npy does not exist! Return to main menu")
        raise IOError
    
    try:
        y = torch.Tensor(np.load(f"{path}/{name}/y_data_prep.npy")).to(params.device)
    except:
        print(f"{path}/{name}/y_data_prep.npy does not exist! Return to main menu")
        raise IOError

    if trainSetBool:
        params.training_size = np.load(f"{path}/{name}/y_data_prep.npy").shape[0]

    dataset = models.dataset.dataset(z,y)    
    return DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)

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
    state_dict = torch.load(filepath + "/state.pth", map_location=torch.device(params.device))
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
        if input == 'True':
            return True
        elif input == 'False':
            return False
        else:
            raise TypeError("Input must be True or False")
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


#create and load program configuration
config = Settings().load()
config.create_dirs()