"""
@author Todd

"""


import numpy as np

class LM_Experiments():

    def __init__(self,ll):
        self.ll=ll

    def combination1(self):
       x_inp = 10
       seq_len = 11
       epochs_no = 500
       y_pred = self.ll.learnLinearFunc(x_inp, seq_len, epochs_no)
       print(y_pred)

    def combination2(self):
       x_inp = 2600
       seq_len = 2000
       epochs_no = 1
       y_pred = self.ll.learnLinearFunc(x_inp, seq_len, epochs_no)
       print(y_pred)
