import os
from pd4ml import TopTagging
import numpy as np
from utilities import config

def load(number_of_events:int=10000):
    
    PATH = f"{config.path_merged}/TopTagging"
    
    print(f"Loading pd4ml dataset with {number_of_events} events:")
    
    X_train_f, y_train_f = TopTagging.load('train', path = config.path_merged)
    X_test_f, y_test_f = TopTagging.load('test', path = config.path_merged)

    y_train_ft = []
    y_test_ft = []

    for y in y_train_f:
        if y == 0:
            y_train_ft.append([0,1])
        else:
            y_train_ft.append([1,0])

    for y in y_test_f:
        if y == 0:
            y_test_ft.append([0,1])
        else:
            y_test_ft.append([1,0])

    X_train_f = X_train_f[0]
    X_test_f = X_test_f[0]

    X_train = X_train_f[ 0:number_of_events ]
    y_train = y_train_ft[ 0:number_of_events ]
    X_test = X_test_f[ 0:number_of_events ]
    y_test = y_test_ft[ 0:number_of_events ]
    X_val = X_train_f[ -number_of_events: ]
    y_val = y_train_ft[ -number_of_events: ]

    print(f"Saving to {PATH}")
    
    for set in [["test", X_test, y_test], ["train", X_train, y_train], ["val", X_val, y_val]]:
        if not os.path.exists(f"{PATH}/{set[0]}"):
            os.makedirs(f"{PATH}/{set[0]}", exist_ok=True)
        print(f"Saving to {set[0]} data to {PATH}/{set[0]}")
        np.save(f"{PATH}/{set[0]}/x_data.npy",set[1])
        np.save(f"{PATH}/{set[0]}/y_data.npy",set[2])
    
