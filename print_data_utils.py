from os import system, name
import time
import numpy as np

# a useful function to present things clearer
def separator():
    print( "-----------------------------------------------" )


def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def formatTime(time):
    
    time = round(time)
    
    minutes = int((time - time%60)/60)
    seconds = time%60

    return minutes, seconds


