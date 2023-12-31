import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import functional as F

import numpy as np

'''
Custom LSTM in PyTorch as seen in:
https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf
http://www.bioinf.jku.at/publications/older/2604.pdf
'''


class LSTMCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(LSTMCell, self).__init__()

        print("Initializing LSTMCell")
        self.hidden_size = hidden_size

        #Initialize weights for RNN cell
        if(weight_init==None):
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = nn.init.xavier_normal_(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = nn.init.xavier_normal_(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = nn.init.xavier_normal_(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = nn.init.xavier_normal_(self.W_c)
        else:
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = weight_init(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = weight_init(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = weight_init(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = weight_init(self.W_c)

        if(reccurent_weight_init == None):
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = nn.init.orthogonal_(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = nn.init.orthogonal_(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = nn.init.orthogonal_(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = nn.init.orthogonal_(self.U_c)
        else:
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = recurrent_weight_initializer(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = recurrent_weight_initializer(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = recurrent_weight_initializer(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = recurrent_weight_initializer(self.U_c)

        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        #Set up dropout layer if requested
        if(drop==0):
            self.keep_prob = False
        else:
            self.keep_prob = True
            self.dropout = nn.Dropout(drop)
        if(rec_drop == 0):
            self.rec_keep_prob = False
        else:
            self.rec_keep_prob = True
            self.rec_dropout = nn.Dropout(rec_drop)

        #Initialize recurrent states h_t and c_t
        self.states = None

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda(), Variable(torch.randn(batch_size, self.hidden_size)).cuda())
        else:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)), Variable(torch.randn(batch_size, self.hidden_size)))
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self, X_t):
    ############################################# FOR STUDENTS #####################################
        h_t_previous, c_t_previous = self.states

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)

        # Forget Gate Equation
        # Instead of using np.dot
        f_t = torch.sigmoid(torch.mm(X_t,self.W_f) + torch.mm(h_t_previous,self.U_f) + self.b_f)


        # Input Gate Equation
        i_t = torch.sigmoid(torch.mm( X_t,self.W_i) + torch.mm( h_t_previous,self.U_i) + self.b_i)

        # Output Gate Equation
        o_t = torch.sigmoid(torch.mm( X_t,self.W_o) + torch.mm( h_t_previous,self.U_o,) + self.b_o)

        # Cell State Equation
        c_hat_t = torch.tanh(torch.mm( X_t,self.W_c) + torch.mm( h_t_previous,self.U_c) + self.b_c)
        c_t = f_t * c_t_previous + i_t * c_hat_t

        # Define Hidden Output Equation
        h_t = o_t * torch.tanh(c_t)

    #################################################################################################
        self.states = (h_t, c_t)
        return h_t


class LSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1,
                 drop=None,
                 rec_drop=None):
        super(LSTM, self).__init__()
        #Initialize deep RNN neural network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        #Initialize individual LSTM cells
        self.lstms = nn.ModuleList()
        self.lstms.append(LSTMCell(input_size=input_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))

        for index in range(self.layers-1):
            self.lstms.append(LSTMCell(input_size=hidden_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))

        #Initialize weights for output linear layer
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states for all RNN cells defined
        for index in range(len(self.lstms)):
            self.lstms[index].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        #Define forward method for deep RNN neural network
        for index in range(len(self.lstms)):
            x = self.lstms[index](x)
        out = self.fc1(x)
        return out
