#Imports
import pickle
import numpy as np
from os import listdir
import re
from FromKeras import parse_stories, get_stories #this file used from Keras examples https://keras.io/examples/babi_memnn/
#Get training and test data

training1KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt'
training10KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt'
testing1KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en/qa6_yes-no-questions_test.txt'

f=open(training1KPath,'rb')
training_data1K=get_stories(f)
f=open(testing1KPath,'rb')
testing_data=get_stories(f)
f=open(training10KPath,'rb')
training_data10K=get_stories(f)

print('length of training data 1K', len(training_data1K))
print('length of training data 10K', len(training_data10K))
print('length of testing data',len(testing_data))
#Training and test data now in memory. Here pick either 10k or 1k as training data

training_data=training_data10K

#Create a vocab from the training data
vocab = set()
for story, question, answer in training_data:
    vocab = vocab.union(story)
    vocab = vocab.union(question)
    vocab.add(answer)
print('the length of the vocab is',len(vocab))

#Determine the longest sentence length from both training and testing data

longest_train_sentence=0
longest_train_question=0

for story,question,answer in training_data:

    if len(story)>longest_train_sentence:
        longest_train_sentence=len(story)
    if len(question)>longest_train_question:
        longest_train_question=len(question)

print('the longest training story is',longest_train_sentence, 'the longest training question is',longest_train_question)

longest_test_sentence=0
longest_test_question=0

for story,question,answer in testing_data:
    if len(story)>longest_test_sentence:
        longest_test_sentence=len(story)
    if len(question)>longest_test_question:
        longest_test_question=len(question)
print('the longest testing story is',longest_test_sentence,'longest test question ',longest_test_question)
longest_story_sentence = max(longest_train_sentence,longest_test_sentence)
longest_question_sentence=max(longest_train_question,longest_test_question)








