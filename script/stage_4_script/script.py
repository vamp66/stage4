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


''' heres the output i got after running this file
[nltk_data] Downloading package punkt to
[nltk_data]     /Users/annaghiuzeli/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /Users/annaghiuzeli/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/annaghiuzeli/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Text Classification
loading data...
Classification: rnn
evaluating performance...
Epoch: 0 Loss: 0.6935764965653419 Test Accuracy: 0.60852
evaluating performance...
Epoch: 1 Loss: 0.6472537857234478 Test Accuracy: 0.65096
evaluating performance...
Epoch: 2 Loss: 0.5963112707495689 Test Accuracy: 0.65168
evaluating performance...
Epoch: 3 Loss: 0.5794219510793686 Test Accuracy: 0.6794
evaluating performance...
Epoch: 4 Loss: 0.5506578692853451 Test Accuracy: 0.72976
evaluating performance...
Epoch: 5 Loss: 0.5475578138470649 Test Accuracy: 0.7722
evaluating performance...
Epoch: 6 Loss: 0.5007258998692036 Test Accuracy: 0.77936
evaluating performance...
Epoch: 7 Loss: 0.5047449282020331 Test Accuracy: 0.76432
evaluating performance...
rnn Accuracy: 0.76432
rnn Precision: 0.833939761471599
rnn Recall: 0.66008
rnn F1: 0.7368938108421899
Classification: lstm
evaluating performance...
Epoch: 0 Loss: 0.6374562740921974 Test Accuracy: 0.6514
evaluating performance...
Epoch: 1 Loss: 0.5296015695273876 Test Accuracy: 0.76664
evaluating performance...
Epoch: 2 Loss: 0.44213932681679724 Test Accuracy: 0.78616
evaluating performance...
Epoch: 3 Loss: 0.38405142399370673 Test Accuracy: 0.8086
evaluating performance...
Epoch: 4 Loss: 0.35139531893730164 Test Accuracy: 0.83632
evaluating performance...
Epoch: 5 Loss: 0.30948184695988895 Test Accuracy: 0.82984
evaluating performance...
Epoch: 6 Loss: 0.28293418326824904 Test Accuracy: 0.82108
evaluating performance...
Epoch: 7 Loss: 0.2568263270556927 Test Accuracy: 0.851
evaluating performance...
lstm Accuracy: 0.851
lstm Precision: 0.8930395055092717
lstm Recall: 0.79752
lstm F1: 0.8425812449816169
Classification: gru
evaluating performance...
Epoch: 0 Loss: 0.6443904952466488 Test Accuracy: 0.70816
evaluating performance...
Epoch: 1 Loss: 0.544744626313448 Test Accuracy: 0.7778
evaluating performance...
Epoch: 2 Loss: 0.45078331515789033 Test Accuracy: 0.82068
evaluating performance...
Epoch: 3 Loss: 0.3698633482426405 Test Accuracy: 0.83088
evaluating performance...
Epoch: 4 Loss: 0.31695181863904 Test Accuracy: 0.84956
evaluating performance...
Epoch: 5 Loss: 0.2848575001329184 Test Accuracy: 0.85512
evaluating performance...
Epoch: 6 Loss: 0.25770629004240037 Test Accuracy: 0.85736
evaluating performance...
Epoch: 7 Loss: 0.23622231555134057 Test Accuracy: 0.85964
evaluating performance...
gru Accuracy: 0.85964
gru Precision: 0.8610553369207292
gru Recall: 0.85768
gru F1: 0.8593643541341028
Text Generation
loading data...
Generation: rnn
Epoch: 0 Loss: 7.162637612969602
Epoch: 1 Loss: 5.856036065686058
Epoch: 2 Loss: 5.5341795237525915
Epoch: 3 Loss: 5.252540506358211
Epoch: 4 Loss: 5.001132165459611
Epoch: 5 Loss: 4.776959616511729
Epoch: 6 Loss: 4.569953289337723
Epoch: 7 Loss: 4.379438090412596
Epoch: 8 Loss: 4.197236852786985
Epoch: 9 Loss: 4.022480530450729
Epoch: 10 Loss: 3.859878161832525
Epoch: 11 Loss: 3.6999595003563144
Epoch: 12 Loss: 3.5441719972690438
Epoch: 13 Loss: 3.400238124862699
Epoch: 14 Loss: 3.257534775869179
Epoch: 15 Loss: 3.115271271818516
Epoch: 16 Loss: 2.9797803849974045
Epoch: 17 Loss: 2.8551416169259465
Epoch: 18 Loss: 2.730433426420727
Epoch: 19 Loss: 2.6124866396813444
Epoch: 20 Loss: 2.4995523848457197
Epoch: 21 Loss: 2.3926448075309783
Epoch: 22 Loss: 2.293248924756315
Epoch: 23 Loss: 2.195756177572963
Epoch: 24 Loss: 2.100149845665392
Epoch: 25 Loss: 2.011939834357778
Epoch: 26 Loss: 1.93060615706532
Epoch: 27 Loss: 1.8455902793698482
Epoch: 28 Loss: 1.7705992029568711
Epoch: 29 Loss: 1.7003204927638484
Epoch: 30 Loss: 1.624352250454899
Epoch: 31 Loss: 1.5580515782871316
Epoch: 32 Loss: 1.495016227227104
Epoch: 33 Loss: 1.4328830583174925
Epoch: 34 Loss: 1.3743688142990507
Epoch: 35 Loss: 1.31901695666566
Epoch: 36 Loss: 1.2678652370461112
Epoch: 37 Loss: 1.224555308883493
start: what's the loudest
generated: what's the loudest economic system ? capitalism
original: what's the loudest economic system ? capitalism
exact match: True, word overlap: 1.0

start: why don't blind
generated: why don't blind people like to skydive ? because it scares their dogs ?
original: why don't blind people like to go skydiving ? it scares their seeing - eye dog .
exact match: False, word overlap: 0.79

start: my dental hygienist
generated: my dental hygienist retired after working 55 years . . . but i can see your drawers !
original: my dental hygienist retired after working 55 years . . . all she got was a lousy plaque . . .
exact match: False, word overlap: 0.61

Generation: lstm
Epoch: 0 Loss: 7.42673045903922
Epoch: 1 Loss: 6.047954614594597
Epoch: 2 Loss: 5.688977112752442
Epoch: 3 Loss: 5.463701209069474
Epoch: 4 Loss: 5.264036365560774
Epoch: 5 Loss: 5.085050173958192
Epoch: 6 Loss: 4.917071116102903
Epoch: 7 Loss: 4.7651044875567115
Epoch: 8 Loss: 4.617363623420348
Epoch: 9 Loss: 4.4778443444083855
Epoch: 10 Loss: 4.34238246278022
Epoch: 11 Loss: 4.212137127628162
Epoch: 12 Loss: 4.08818266100713
Epoch: 13 Loss: 3.9665743177239903
Epoch: 14 Loss: 3.8517885840189003
Epoch: 15 Loss: 3.736921044959975
Epoch: 16 Loss: 3.6292454796566473
Epoch: 17 Loss: 3.5206736038116286
Epoch: 18 Loss: 3.4155150863598074
Epoch: 19 Loss: 3.3141289086582923
Epoch: 20 Loss: 3.2155032102041563
Epoch: 21 Loss: 3.1180355532689688
Epoch: 22 Loss: 3.022807031612655
Epoch: 23 Loss: 2.92846007779259
Epoch: 24 Loss: 2.8393351817101644
Epoch: 25 Loss: 2.7489127418997845
Epoch: 26 Loss: 2.662577359802773
Epoch: 27 Loss: 2.5753015210684365
Epoch: 28 Loss: 2.4929659213737554
Epoch: 29 Loss: 2.4167634492737737
Epoch: 30 Loss: 2.339264287166619
Epoch: 31 Loss: 2.263411866751317
Epoch: 32 Loss: 2.1925857187640356
Epoch: 33 Loss: 2.122290398277866
Epoch: 34 Loss: 2.054805026836372
Epoch: 35 Loss: 1.9908794005907862
Epoch: 36 Loss: 1.9274993938670646
Epoch: 37 Loss: 1.8684402341172375
start: what's the loudest
generated: what's the loudest economic system ? a stick
original: what's the loudest economic system ? capitalism
exact match: False, word overlap: 0.75

start: why don't blind
generated: why don't blind people like to skydive ? because it scares their seeing .
original: why don't blind people like to go skydiving ? it scares their seeing - eye dog .
exact match: False, word overlap: 0.86

start: my dental hygienist
generated: my dental hygienist retired after working 55 years . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
original: my dental hygienist retired after working 55 years . . . all she got was a lousy plaque . . .
exact match: False, word overlap: 1.0

Generation: gru
Epoch: 0 Loss: 7.317589208318331
Epoch: 1 Loss: 5.913181096499122
Epoch: 2 Loss: 5.5611599588217775
Epoch: 3 Loss: 5.263154958238143
Epoch: 4 Loss: 4.991729458516094
Epoch: 5 Loss: 4.7471557319090785
Epoch: 6 Loss: 4.5218905391293305
Epoch: 7 Loss: 4.317899541114026
Epoch: 8 Loss: 4.121305604751249
Epoch: 9 Loss: 3.9377503042303386
Epoch: 10 Loss: 3.758359235400189
Epoch: 11 Loss: 3.590279770250238
Epoch: 12 Loss: 3.425298389024417
Epoch: 13 Loss: 3.266890964437501
Epoch: 14 Loss: 3.1186237486605286
Epoch: 15 Loss: 2.9757325464053452
Epoch: 16 Loss: 2.8360791792851927
Epoch: 17 Loss: 2.7013661130876927
Epoch: 18 Loss: 2.573062415334652
Epoch: 19 Loss: 2.450777868748299
Epoch: 20 Loss: 2.3371983288836686
Epoch: 21 Loss: 2.2269581465186086
Epoch: 22 Loss: 2.1242488503603107
Epoch: 23 Loss: 2.0252026358456265
Epoch: 24 Loss: 1.929357607032456
Epoch: 25 Loss: 1.8402530681778264
Epoch: 26 Loss: 1.753056095648929
Epoch: 27 Loss: 1.6726809208695896
Epoch: 28 Loss: 1.596587358435338
Epoch: 29 Loss: 1.5211340783556646
Epoch: 30 Loss: 1.4508460493328832
Epoch: 31 Loss: 1.3842338106782164
Epoch: 32 Loss: 1.323110800342113
Epoch: 33 Loss: 1.2624831282695625
Epoch: 34 Loss: 1.2058965529966295
Epoch: 35 Loss: 1.1539927385885236
Epoch: 36 Loss: 1.105067589537283
Epoch: 37 Loss: 1.0580814975037087
start: what's the loudest
generated: what's the loudest economic system ? capitalism
original: what's the loudest economic system ? capitalism
exact match: True, word overlap: 1.0

start: why don't blind
generated: why don't blind people like to skydive ? because it scares the crap out of the gutter !
original: why don't blind people like to go skydiving ? it scares their seeing - eye dog .
exact match: False, word overlap: 0.5

start: my dental hygienist
generated: my dental hygienist retired after working 55 years . . . i don't know where he was too modest .
original: my dental hygienist retired after working 55 years . . . all she got was a lousy plaque . . .
exact match: False, word overlap: 0.65
'''
