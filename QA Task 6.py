#Imports
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from FromKeras import parse_stories, get_stories #this file used from Keras examples https://keras.io/examples/babi_memnn/
from keras.models import load_model
import pickle
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
length_vocab=len(vocab)+1
print('the length of the vocab is',length_vocab)

#Determine the longest sentence length from both training and testing data

longest_train_sentence=0
longest_train_question=0
story_text=[]
for story,question,answer in training_data:
    story_text.append(story)

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

#Padding
t = Tokenizer(filters=[])  # ensure that no filters are used. By default this will filter out punctuation.

#Here try to see if there is already a tokenizer saved that has been previously fit to the vocab. If so load
#that one. The reason is that previously saved keras models will have been trained on that vocab and we want
#the same results
try:
    with open('tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)
except:
    t.fit_on_texts(vocab)
    #print(t.index_word)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Tokenize Story, question, answers

story_1_indexes=[]
story_1_indexes.append(story_text[0])
story_1_indexes = t.texts_to_sequences(story_1_indexes)
#story_1_indexes=pad_sequences(story_1_indexes,maxlen=longest_story_sentence)
#print(story_text[0])
#print(story_1_indexes[0])


stories=[]
questions=[]
answers=[]

for story, question, answer in training_data:
    stories.append(story)
    questions.append(question)
    answers.append(answer)


def toSequence(input, t):
    sequenced = []
    for x in input:
        sequenced.append([val for sublist in t.texts_to_sequences(x) for val in sublist])
    return sequenced


story_sequences=toSequence(stories,t)
question_sequences=toSequence(questions,t)
answer_sequences=toSequence(answers,t)



#print(story_sequences[0])
#print(question_sequences[0])


# Create a function for vectorizing the stories, questions and answers: ****************************TAKEN******
import numpy as np

def vectorize_stories(data, word_index=t.word_index, max_story_len=longest_story_sentence,
                      max_question_len=longest_question_sentence):
    # vectorized stories:
    X = []
    # vectorized questions:
    Xq = []
    # vectorized answers:
    Y = []

    for story, question, answer in data:
        # Getting indexes for each word in the story
        x = [word_index[word.lower()] for word in story]
        # Getting indexes for each word in the story
        xq = [word_index[word.lower()] for word in question]
        # For the answers
        y = np.zeros(len(word_index) + 1)  # Index 0 Reserved when padding the sequences
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # Now we have to pad these sequences:
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, questions_train, answers_train = vectorize_stories(training_data)
inputs_test, questions_test, answers_test = vectorize_stories(testing_data)



print(stories[0])
print(questions[0])
print(answers[0])

print(inputs_train[0])
print(questions_train[0])
print(answers_train[0])

#*******************************************************************************************/TAKEN

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

# We need to create the placeholders
#The Input function is used to create a keras tensor
#PLACEHOLDER shape = (max_story_len,batch_size)
#These are our placeholder for the inputs, ready to recieve batches of the stories and the questions
input_sequence = Input((longest_story_sentence,)) #As we dont know batch size yet
question = Input((longest_question_sentence,))


#Create input encoder M:
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=length_vocab,output_dim = 64)) #From paper
input_encoder_m.add(Dropout(0.3))

#Outputs: (Samples, story_maxlen,embedding_dim) -- Gives a list of the lenght of the samples where each item has the
#lenght of the max story lenght and every word is embedded in the embbeding dimension

#Create input encoder C:
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=length_vocab,output_dim = longest_question_sentence)) #From paper
input_encoder_c.add(Dropout(0.3))

#Outputs: (samples, story_maxlen, max_question_len)

#Create question encoder:
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=length_vocab,output_dim = 64,input_length=longest_question_sentence)) #From paper
question_encoder.add(Dropout(0.3))

#Outputs: (samples, question_maxlen, embedding_dim)

#Now lets encode the sequences, passing the placeholders into our encoders:
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#Use dot product to compute similarity between input encoded m and question
#Like in the paper:
match = dot([input_encoded_m,question_encoded], axes = (2,2))
match = Activation('softmax')(match)

#For the response we want to add this match with the ouput of input_encoded_c
response = add([match,input_encoded_c])
response = Permute((2,1))(response) #Permute Layer: permutes dimensions of input

#Once we have the response we can concatenate it with the question encoded:
answer = concatenate([response, question_encoded])

# Reduce the answer tensor with a RNN (LSTM)
answer = LSTM(32)(answer)

#Regularization with dropout:
answer = Dropout(0.5)(answer)
#Output layer:
answer = Dense(length_vocab)(answer) #Output shape: (Samples, Vocab_size) #Yes or no and all 0s

#Now we need to output a probability distribution for the vocab, using softmax:
answer = Activation('softmax')(answer)

#Now we build the final model:
model = Model([input_sequence,question], answer)

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#Categorical instead of binary cross entropy as because of the way we are training
#we could actually see any of the words from the vocab as output
#however, we should only see yes or no


try:
    model = load_model('300Epochs.h5')
    print("Successfully loaded saved model")
except:
    history = model.fit([inputs_train,questions_train],answers_train, batch_size = 32, epochs = 800, validation_data = ([inputs_test,questions_test],answers_test))
    model.save('300Epochs.h5')


#model.summary()
index=25
print(testing_data[index])
pred_results = model.predict(([inputs_test,questions_test]))

val_max= np.argmax(pred_results[index])
#print(val_max)
for key,val in t.word_index.items():
    if val == val_max:
        k = key
print(k)

print(pred_results[index][val_max])

print(vocab)

my_story = 'John put down the apple . Sandra picked up the milk . John journeyed to the office . '
my_question = 'John in the office ?'
my_data = [(my_story.split(), my_question.split(),'yes')]
my_story, my_ques, my_ans = vectorize_stories(my_data)
pred_results = model.predict(([my_story,my_ques]))
val_max = np.argmax(pred_results[0])
for key,val in t.word_index.items():
    if val == val_max:
        k = key
print(k)
print(pred_results[0][val_max])



def input_story():
    story=[]
    for i in range(3):
        story.append(input("Enter Story Sentence :"))
    question=input("What is the question?:")
    return story,question

#story,question=input_story()
#print(story, question)