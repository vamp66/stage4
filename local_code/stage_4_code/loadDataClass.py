#Anna Ghiuzeli, student ID: 921637976 
#I referred to the given Dataset_Loader.py from the source code template but I modified it to work with my stage 4 here
#also I refered to the given website in the READme for text classification from stage4 data folder for like cleaning which was https://machinelearningmastery.com/clean-text-machine-learning-python/ 

#imported modules used and also the given dataset.py 
import os
import string
import nltk
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from local_code.base_class.dataset import dataset

#this downloads the NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class dataClass(dataset):
    dataset_source_folder_path = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.data = None
        self.max_vocab = 10000
        self.max_len = 150
        self.vocab = None
        
    #this justs cleans the text by gettinf rid of line breaks, splitting text into works, changing it all to lowercase, removing punctuation & stop words, keep only words made of letters so no like digits in there
    def clean(self, text):
        text = text.replace('<br />', ' ')
        words = word_tokenize(text)
        words = [word.lower() for word in words]
        table = str.maketrans('', '', string.punctuation)
        words = [word.translate(table) for word in words]
        words = [word for word in words if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return words
        
    # this reads the pos and neg reviews
    def readIn(self, split):
        X = []
        y = []
        for label, folder_name in [(1, 'pos'), (0, 'neg')]:
            folder = os.path.join(
                self.dataset_source_folder_path, split, folder_name)
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    review = f.read()
                X.append(self.clean(review))
                y.append(label)
        return X, y
        
    #this encodes the words
    def encode(self, words):
        X = []
        for word in words[:self.max_len]:
            if word in self.vocab:
                X.append(self.vocab[word])
            else:
                X.append(1)
        #i padded the beginning with 0's to make them all the same len
        while len(X) < self.max_len:
            X.insert(0, 0)
        return X
        
    #this load the training and testing data
    def load(self):
        if self.data is not None:
            return self.data
        print('loading data...')
        train_X, train_y = self.readIn('train')
        test_X, test_y = self.readIn('test')
        #this counts words in training reviews
        word_counts = Counter()
        for review in train_X:
            word_counts.update(review)
        #this makes vocab using most common words
        self.vocab = {'<pad>': 0,'<unk>': 1}
        for word, count in word_counts.most_common(self.max_vocab - 2):
            self.vocab[word] = len(self.vocab)
        #encodes training and testing reviews
        X_train = []
        X_test = []
        for review in train_X:
            X_train.append(self.encode(review))
        for review in test_X:
            X_test.append(self.encode(review))
        #this stores the data
        self.data = {
            'train': { 'X': X_train, 'y': train_y},
            'test': { 'X': X_test, 'y': test_y}}
        return self.data