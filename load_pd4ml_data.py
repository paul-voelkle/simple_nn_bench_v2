import os
from pd4ml import TopTagging
import numpy as np

def load(number_of_events:int=10000):
    PATH="data/not_processed/TopTagging"
    
    print(f"Loading pd4ml dataset with {number_of_events} events:")
    
    X_train_f, y_train_f = TopTagging.load('train', path = 'data/not_processed')
    X_test_f, y_test_f = TopTagging.load('test', path = 'data/not_processed')

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
    
    for paths in ["test", "train", "val"]:
        if not os.path.exists(f"{PATH}/{paths}"):
            os.makedirs(f"{PATH}/{paths}", exist_ok=True)
    
    np.save(f"{PATH}/train/x_data.npy",X_train)
    np.save(f"{PATH}/train/y_data.npy",y_train)
    np.save(f"{PATH}/test/x_data.npy",X_test)
    np.save(f"{PATH}/test/y_data.npy",y_test)
    np.save(f"{PATH}/val/x_data.npy",X_val)
    np.save(f"{PATH}/val/y_data.npy",y_val)