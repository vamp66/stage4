#Anna Ghiuzeli, student ID: 921637976 
#I used the given Method_MLP.py from source code template but I modified it to work with my stage 4 here

#imported modules used and also the given method.py 
from local_code.base_class.method import method 
import torch
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class gen(method, nn.Module):
    #i just used arbitrary vals except for the learning rate which i just used same as in the given Method_MLP.py
    data = None
    max_epoch = 38
    batch_size = 47
    learning_rate = 1e-3
    embed_dim = 116
    hidden_dim = 199
    curve_file = 'curve.png'
    #this initializes the model
    def __init__(self, mName, mDescription, vocab_size, rnn_type='rnn'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        #this embeds each word
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        #this chooses btwn diff rnn type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(self.embed_dim, self.hidden_dim, batch_first=True)
        self.fc_layer = nn.Linear(self.hidden_dim, vocab_size)
        self.loss_list = []

    #this is for forward propagation & it turns the reviews' word encodings into embedding then runs them throuhg the rnn type chosen earlier then predicts next word
    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        y_pred = self.fc_layer(output)
        return y_pred

    #trains model
    def train(self, X):
        X = torch.LongTensor(np.array(X))
        #i used adam for the optimizer and cross entropy as loss fucnt
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(self.max_epoch): 
            #i shuffled the data so it wouldnt be ordered which could cause the model to learn just the order
            order = torch.randperm(len(X))
            total_loss = 0
            #used smaller batches
            for i in range(0, len(X), self.batch_size):
                batch = X[order[i:i + self.batch_size]]
                #i used all words except the last one as input bc the next word is the correct answer
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                #this just gets the pred then calculates the loss & grad
                y_pred = self.forward(inputs)
                train_loss = loss_function(
                    y_pred.reshape(-1, y_pred.shape[-1]),
                    targets.reshape(-1))
                optimizer.zero_grad()
                train_loss.backward()
                 #this stops the gradient from getting too big so training stays stable
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                #updates the model
                optimizer.step()
                total_loss += train_loss.item() * len(batch)
            avg_loss = total_loss / len(X)
            self.loss_list.append(avg_loss)
            #prints teh loss & test accuracy for all epochs
            print('Epoch:', epoch, 'Loss:', avg_loss)

    #this generates a joke from trained model
    def generate(self, start_words, max_words=50):
        wordNum = self.data['wordNum']
        Numword = self.data['Numword']
        #encode starting words into numbers
        ids = []
        for word in start_words:
            ids.append(wordNum.get(word, 1))
        with torch.no_grad():
            for i in range(max_words):
                #predicts then picks word with the highest prediction then stops when the model predicts the end token
                y_pred = self.forward(torch.LongTensor([ids]))
                next_id = y_pred[0, -1].argmax().item()
                if next_id == wordNum['<eos>']:
                    break
                ids.append(next_id)
        #decodes nums back to words
        joke = []
        for number in ids:
            joke.append(Numword[number])
        return ' '.join(joke)

    #this plots the training loss
    def plotGraph(self):
        plt.plot(self.loss_list)
        plt.title(self.rnn_type + ' training loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig(self.curve_file)
        plt.close()

    #this runs the model & saves the training loss graph
    def run(self):
        self.train(self.data['X'])
        self.plotGraph()