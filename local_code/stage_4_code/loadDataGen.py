#Anna Ghiuzeli, student ID: 921637976 
#I used the given Dataset_Loader.py from source code template but I modified it to work with my stage 4 here
#also i referred to the given website in the NLP & Word Embeddings lecture slides which was https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html 

#imported modules used and also the given dataset.py 
import csv
import re
from local_code.base_class.dataset import dataset

class loadGen(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.data = None
        #makes joke max len be 50
        self.max_len = 50
    
    #this makes tokens for the word itself & punctuation from the joke
    def tokenize(self, text):
        words = re.findall(r"[\w']+|[^\w\s]", text.lower())
        return words

    #this loads the jokes & encodes the words
    def load(self):
        if self.data is not None:
            return self.data
        print('loading data...')
        jokes = []
        f = open(
            self.dataset_source_folder_path + self.dataset_source_file_name, 'r',encoding='utf-8')
        reader = csv.reader(f)
        next(reader)
        #reads in the joke 
        for row in reader:
            joke = self.tokenize(row[-1])
            #keeps track of the end of the joke
            joke.append('<eos>')
            jokes.append(joke)
        f.close()
        #gives each word a distinct number
        wordNum= { '<pad>': 0,'<unk>': 1, '<eos>': 2}
        for joke in jokes:
            for word in joke:
                if word not in wordNum:
                    wordNum[word] = len(wordNum)
        #this stores the decoding conversion
        Numword = {}
        for word in wordNum:
            number = wordNum[word]
            Numword[number] = word
        #encodes each joke via number list
        X = []
        for joke in jokes:
            numbers = []
            for word in joke:
                numbers.append(wordNum[word])
            numbers = numbers[:self.max_len]
            #pads w zeros at the end if jokes' len short
            while len(numbers) < self.max_len:
                numbers.append(0)
            X.append(numbers)
        #this stores the data
        self.data = { 'X': X, 'jokes': jokes, 'wordNum': wordNum, 'Numword': Numword}
        return self.data