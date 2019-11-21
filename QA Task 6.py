#Imports
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
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
t=Tokenizer(filters=[]) #ensure that no filters are used. By default this will filter out punctuation.
t.fit_on_texts(vocab)
print(t.index_word)

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

print(stories[0])
print(questions[0])
print(answers[0])



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

#*******************************************************************************************/TAKEN

