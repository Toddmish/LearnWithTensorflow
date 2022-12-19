"""
@author Todd

"""


import numpy as np
from todProj.Teask1.experiments import LM_Experiments
from todProj.Teask1.learn import Learn


def print_hi(text):
    # Use a breakpoint in the code line below to debug your script.

    print(f'Hi, {text}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('Using tensorflow to learn mathematical functions')
    print(__name__)

    learn=Learn()
    experiments= LM_Experiments(learn)

    experiments.combination1()



