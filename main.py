import os 
import models
from load_pd4ml_data import load
from preprocess_data import preprocess_data, merge_data
from training_network import train_network
from test_networks import test_model
from time import sleep
from utilities import Settings, clear, separator, argument_handler, invalid_args_error, max_arg_error, min_arg_error, input_handler, command_handler

config = Settings()
config.load()
config.create_dirs()


#some drawing utilities
def start_up():
    clear()
    separator()
    print("Simple NN-Bench v0.0.3")
    separator()

def returning():
    print("Done, press enter to return to main menu")
    input()
    start_up()


#show functions
def show(args:list[str]=[]):
    allowed_args = ["models", "datasets", "config"]
    functions = [show_models, show_datasets, show_config]
    if len(args) == 0:
        print("Usage:")
        print("show models [available/trained]")
        print("show datasets [not_merged/not_preprocessed/preprocessed]")
        print("show config")
    elif argument_handler(args, allowed_args, functions):
        return
    else:
        invalid_args_error(args)

def show_models(args:str=list[str]):
    if max_arg_error(args,1) or min_arg_error(args,1):
        print("Usage:show models [available/trained]")
    elif args[0] == "":
        print("Usage:show models [available/trained]")
    elif args[0] == "available":
        separator()
        print("Available Models:")
        separator()
        for name in models.__models__:
            print(name)
        separator()
    elif args[0] == "trained":
        separator()
        print("Trained Models:")
        separator()
        try:
            print(os.listdir("trained_models/"))
        except:
            print("Directory trained_models/ does not exist!")
        separator()
    elif args[0] in models.__models__:
        model_class = getattr(models, args[0])
        print(model_class.Model())    
    else:
        invalid_args_error(args)
        print("Usage:show models [available/trained]")

def show_datasets(args:str=list[str]):
    if max_arg_error(args,1) or min_arg_error(args,1):
        print("Usage:show datasets [not_merged/not_processed/preprocessed]")
    elif args[0] in ["not_merged", "not_processed", "preprocessed"]:
        separator()
        print(f"Available datasets in data/{args[0]}:")
        separator()
        try:
            print(os.listdir(f"data/{args[0]}/"))
        except:
            print(f"Directory data/{args[0]}/ does not exist!")
        separator()
    else:
        invalid_args_error([args])
        print("Usage:show datasets [not_merged/not_processed/preprocessed]")

def show_help(args:list[str]):
    if args == []:
        f = open('help.txt', 'r')
        print(f.read())

def show_config(args:list[str]):
    print("Program configuration:")
    config.print()


#program mode prompts
def train_prompt(args:list[str]):
    if max_arg_error(args, 2) or min_arg_error(args, 2):
        print("Usage: train modelname dataset")
    elif args[0] in models.__models__:
        print(f"Start training {args[0]}")
        train_network(model_name=args[0], dataset=args[1], config=config)
        returning()
    else:
        print(f"{args[0]} is not a valid model!")
        show(["models", "available"])

def test_prompt(args:list[str]):
    if min_arg_error(args, 2) or max_arg_error(args, 2):
        show(["models", "trained"])
        print("Usage: test modelname test_set")
        return
    #elif args[0] in models.__models__:
    test_model(args[0], args[1], config=config)
    returning()
    #else:
    #    print(f"{args[0]} is not a valid model")
    #   show(["models", "available"])

def preprocess_prompt(args:list[str]):
    if min_arg_error(args, 2):
        print("Usage: preprocess src_folder set_names")
        return
    preprocess_data(src_folder=args[0], files=args[1:], config=config)
    returning()

def load_prompt(args:list[str]):
    if max_arg_error(args, 1) or min_arg_error(args, 1):
        print("Require Number of events!")
        return
    else:
        load(int(args[0]), config=config)
        returning()

def merge_data_prompt(args:list[str]):
    if max_arg_error(args, 3) or min_arg_error(args, 3):
        print("Usage: merge src_data output shuffle[True/False]")
        return
    merge_data(args[0], args[1], args[2]=='True', config=config)
    returning()
    return



#main menu functions
def main():
    while True:
        user_input = input()
        if user_input == "exit":
            config.save()
            print("Bye!")
            sleep(1)
            clear()
            break
        else:
            main_menu_input(user_input)
            
def main_menu_input(input:str=""):
    command, args = input_handler(input)
    commands = ["help", "show", "train", "test", "preprocess", "load", "merge"]
    functions = [show_help, show, train_prompt, test_prompt, preprocess_prompt, load_prompt, merge_data_prompt]
    if command == "":
        print("Try help for commands or exit to close program")
        return
    else:
        command_handler(command, commands, functions, args)


start_up()
main()

