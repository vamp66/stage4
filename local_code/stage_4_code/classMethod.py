#Anna Ghiuzeli, student ID: 921637976 
#I used the given Method_MLP.py from source code template but I modified it to work with my stage 4 here

#imported modules used and also the given method.py and Evaluate_Accuracy.py
from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class methodClass(method, nn.Module):
    #i just used arbitrary vals except for the learning rate which i just used same as in the given Method_MLP.py
    data = None
    max_epoch = 8
    batch_size = 55
    learning_rate = 1e-3
    embed_dim = 99
    hidden_dim = 116
    curve_file = 'curve.png'
    def __init__(self, mName, mDescription, vocab_size, rnn_type='rnn'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        #this embeds each word
        self.embedding = nn.Embedding( vocab_size, self.embed_dim, padding_idx=0)
        #i arbitrarily chose the dropout rate
        self.dropout = nn.Dropout(0.4)
        #this chooses btwn diff rnn type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embed_dim,self.hidden_dim,batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(self.embed_dim,self.hidden_dim,batch_first=True,bidirectional=True)
        #gets the forward & backward states for the output layer
        self.fc_layer = nn.Linear(self.hidden_dim * 2, 2)
        self.loss_list = []
        self.acc_list = []

    #this is for forward propagation & it turns the reviews' word encodings into embedding then runs them throuhg the rnn type chosen earlier then predicts pos or neg
    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        output, h_n = self.rnn(emb)
        if self.rnn_type == 'lstm':
            h_n = h_n[0]
        last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        y_pred = self.fc_layer(self.dropout(last_hidden))
        return y_pred

    #this trains model
    def train(self, X, y):
        X = torch.LongTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        #i used adam for the optimizer and cross entropy as loss fucnt
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('test evaluator', '')
        for epoch in range(self.max_epoch):
            #i shuffled the data so it wouldnt be ordered which could cause the model to learn just the order
            order = torch.randperm(len(X))
            total_loss = 0
            #used smaller batches for the training since before i did this it crashed the program when i tried to run it
            for i in range(0, len(X), self.batch_size):
                j = order[i:i + self.batch_size]
                #this just gets the pred then calculates the loss & grad
                y_pred = self.forward(X[j])
                train_loss = loss_function(y_pred, y[j])
                optimizer.zero_grad()
                train_loss.backward()
                #this stops the gradient from getting too big so training stays stable
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                #this updates model
                optimizer.step()
                total_loss += train_loss.item() * len(j)
            avg_loss = total_loss / len(X)
            pred_y = self.test(self.data['test']['X'])
            accuracy_evaluator.data = {
                'true_y': self.data['test']['y'],'pred_y': pred_y}
            acc = accuracy_evaluator.evaluate()
            self.loss_list.append(avg_loss)
            self.acc_list.append(acc)
            #prints teh loss & test accuracy for all epochs
            print('Epoch:', epoch,'Loss:', avg_loss,'Test Accuracy:', acc)
    
    #this tests the model
    def test(self, X):
        X = torch.LongTensor(np.array(X))
        preds = []
        #this makes dropout not be used for testing
        nn.Module.train(self, False)
        #i used an arbitrary batch size for testing
        with torch.no_grad():
            for i in range(0, len(X), 486):
                y_pred = self.forward(X[i:i + 486])
                preds.append(y_pred.max(1)[1])
        #uses dropout for training
        nn.Module.train(self, True)
        return torch.cat(preds)

    #plots the training loss and testing accuracy 
    def plotGraph(self):
        plt.plot(self.loss_list)
        plt.title(self.rnn_type + ' training loss')
        plt.xlabel('epoch')
        plt.savefig(self.curve_file.replace('.png', '_loss.png'))
        plt.close()
        plt.plot(self.acc_list)
        plt.title(self.rnn_type + ' test accuracy')
        plt.xlabel('epoch')
        plt.savefig(self.curve_file.replace('.png', '_accuracy.png'))
        plt.close()

    #this runs the model & saves the learning curves
    def run(self):
        self.train(self.data['train']['X'],self.data['train']['y'])
        pred_y = self.test(self.data['test']['X'])
        self.plotGraph()
        return {'pred_y': pred_y,'true_y': self.data['test']['y']}