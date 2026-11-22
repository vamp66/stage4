#Anna Ghiuzeli, student ID: 921637976

#here are the modules i used and my local code files imported as well as the given Evalute_Accuracy
import os
import sys
import numpy as np
import torch
#used to let my script run easily from anywhere by finding main project folder
projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(projectRoot)
from local_code.stage_4_code.loadDataClass import dataClass
from local_code.stage_4_code.classMethod import methodClass
from local_code.stage_4_code.loadDataGen import loadGen
from local_code.stage_4_code.genMethod import gen
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import precision_score, recall_score, f1_score

#i used arbitrary val for seed
np.random.seed(6)
torch.manual_seed(6)

#these are the folder paths to get the data and then save to result folder (makes one if it dne)
dataPath = os.path.join(projectRoot, 'stage_4_data', 'text_classification') + '/'
resultPath = os.path.join(projectRoot, 'result', 'stage_4_result') + '/'
os.makedirs(resultPath, exist_ok=True)

print('Text Classification')
#loads IMDB dataset & gets vocab size 
imdbData = dataClass('imdb', '')
imdbData.dataset_source_folder_path = dataPath
data = imdbData.load()
vocabSize = len(imdbData.vocab)

#runs RNN, LSTM, and GRU models on given dataset
for modelType in ['rnn', 'lstm', 'gru']:
    print('Classification:', modelType)
    classifier = methodClass(modelType, '', vocabSize, modelType)
    #saves classification plot and evaluate accuracy 
    classifier.curve_file = resultPath + modelType + '_classification_curve.png'
    classifier.data = data
    accuracyChecker = Evaluate_Accuracy('accuracy', '')
    #trains then gets pred
    predictions = classifier.run()
    #prints accuracy, precision, recall, and f1
    accuracyChecker.data = predictions
    print(modelType, 'Accuracy:', accuracyChecker.evaluate())
    print(modelType, 'Precision:', precision_score(predictions['true_y'], predictions['pred_y']))
    print(modelType, 'Recall:', recall_score(predictions['true_y'], predictions['pred_y']))
    print(modelType, 'F1:', f1_score(predictions['true_y'], predictions['pred_y']))

print('Text Generation')
#loads joke dataset & gets vocab size then picks a few jokes to test 
jokeDataPath = os.path.join(projectRoot, 'stage_4_data', 'text_generation') + '/'
jokeData = loadGen('jokes', '')
jokeData.dataset_source_folder_path = jokeDataPath
jokeData.dataset_source_file_name = 'data'
jokes = jokeData.load()
jokeVocabSize = len(jokes['wordNum'])
testJokes = []
for i in [16, 116, 333]:
    testJokes.append(jokes['jokes'][i])

#runs RNN, LSTM, and GRU models
for modelType in ['rnn', 'lstm', 'gru']:
    print('Generation:', modelType)
    #train then save the generation curve
    jokeModel = gen(modelType, '', jokeVocabSize, modelType)
    jokeModel.curve_file = resultPath + modelType + '_generation_curve.png'
    jokeModel.data = jokes
    jokeModel.run()

    #generate jokes & compare them with actual jokes
    for joke in testJokes:
        #uses first 3 words to write the whole joke
        startWords = joke[:3]
        newJoke = jokeModel.generate(startWords)
        originalJoke = ' '.join(word for word in joke if word != '<eos>')
        #this finds how many generated words appear in the original joke
        generatedWords = newJoke.split(' ')
        overlap = sum(1 for word in generatedWords if word in joke) / len(generatedWords)
        text = ('start: ' + ' '.join(startWords) +'\ngenerated: ' + newJoke +'\noriginal: ' + originalJoke +'\nexact match: ' + str(newJoke == originalJoke) +', word overlap: ' + str(round(overlap, 2)) + '\n')
        print(text)