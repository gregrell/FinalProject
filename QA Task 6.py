#Imports
import pickle
import numpy as np
from os import listdir
import re
from FromKeras import parse_stories, get_stories #this file used from Keras examples https://keras.io/examples/babi_memnn/
#Get training and test data

training1KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt'
training10KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en10-k/qa6_yes-no-questions_train.txt'
testing1KPath='C:/Users/206417559/Documents/GitHub/FinalProject/tasks_1-20_v1-2/en/qa6_yes-no-questions_test.txt'



f=open(training1KPath,'rb')

stories=get_stories(f)

print(len(stories))


