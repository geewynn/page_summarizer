# coding: utf-8

import re
import warnings

from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from attention import AttentionLayer

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

K.clear_session()

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

DATA = pd.read_csv('posts_content.csv')
DATA = DATA.drop(['user_id', 'tags', 'Unnamed: 4'], axis=1)
DATA.head()

DATA.info()

CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not",
                       "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will",
                       "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will",
                       "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",
                       "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have",
                       "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have",
                       "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will",
                       "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}

STOP_WORDS = set(stopwords.words('english'))


def text_cleaner(text, num):
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = ' '.join([CONTRACTION_MAPPING[t] if t in CONTRACTION_MAPPING else
                           t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    new_string = re.sub('[m]{2,}', 'mm', new_string)
    if num == 0:
        tokens = [w for w in new_string.split() if not w in STOP_WORDS]
    else:
        tokens = new_string.split()
    long_words = []

    for i in tokens:
        if len(i) > 1:
            long_words.append(i)
    return (" ".join(long_words)).strip()


CLEANED_TEXT = []
for t in DATA['content']:
    CLEANED_TEXT.append(text_cleaner(t, 0))

CLEANED_SUMMARY = []
for t in DATA['title']:
    CLEANED_SUMMARY.append(text_cleaner(t, 1))

DATA['CLEANED_TEXT'] = CLEANED_TEXT
DATA['CLEANED_SUMMARY'] = CLEANED_SUMMARY

DATA.replace('', np.nan, inplace=True)
DATA.dropna(axis=0, inplace=True)

CNT = 0
for i in DATA['CLEANED_SUMMARY']:
    if len(i.split()) <= 8:
        CNT = CNT + 1
print(CNT / len(DATA['CLEANED_SUMMARY']))

MAX_TEXT_LEN = 30
MAX_SUMMARY_LEN = 8

CLEANED_TEXT = np.array(DATA['CLEANED_TEXT'])
CLEANED_SUMMARY = np.array(DATA['CLEANED_SUMMARY'])

SHORT_TEXT = []
SHORT_SUMMARY = []

for i in range(len(CLEANED_TEXT)):
    if len(CLEANED_SUMMARY[i].split()) <= MAX_SUMMARY_LEN and \
            len(CLEANED_TEXT[i].split()) <= MAX_TEXT_LEN:
        SHORT_TEXT.append(CLEANED_TEXT[i])
        SHORT_SUMMARY.append(CLEANED_SUMMARY[i])

DF = pd.DataFrame({'content': SHORT_TEXT, 'title': SHORT_SUMMARY})

DF['title'] = DF['title'].apply(lambda x: 'sostok ' + x + ' eostok')

X_TR, X_VAL, Y_TR, Y_VAL = train_test_split(np.array(DF['content']),
                                            np.array(DF['title']), test_size=0.1,
                                            random_state=0, shuffle=True)

# prepare a tokenizer for reviews on training DATA
X_TOKENIZER = Tokenizer()
X_TOKENIZER.fit_on_texts(list(X_TR))

THRESH = 4
CNT = 0
TOT_CNT = 0
FREQ = 0
TOT_FREQ = 0

for key, value in X_TOKENIZER.word_counts.items():
    TOT_CNT = TOT_CNT + 1
    TOT_FREQ = TOT_FREQ + value
    if value < THRESH:
        CNT = CNT + 1
        FREQ = FREQ + value

print("% of rare words in vocabulary:", (CNT / TOT_CNT) * 100)
print("Total Coverage of rare words:", (FREQ / TOT_FREQ) * 100)

# prepare a tokenizer for reviews on training DATA
X_TOKENIZER = Tokenizer(num_words=TOT_CNT - CNT)
X_TOKENIZER.fit_on_texts(list(X_TR))

# convert text sequences into integer sequences
X_TR_SEQ = X_TOKENIZER.texts_to_sequences(X_TR)
X_VAL_SEQ = X_TOKENIZER.texts_to_sequences(X_VAL)

# padding zero upto maximum length
X_TR = pad_sequences(X_TR_SEQ, maxlen=MAX_TEXT_LEN, padding='post')
X_VAL = pad_sequences(X_VAL_SEQ, maxlen=MAX_TEXT_LEN, padding='post')

# size of vocabulary ( +1 for padding token)
X_VOC = X_TOKENIZER.num_words + 1

# prepare a tokenizer for reviews on training DATA
Y_TOKENIZER = Tokenizer()
Y_TOKENIZER.fit_on_texts(list(Y_TR))

THRESH = 6
CNT = 0
TOT_CNT = 0
FREQ = 0
TOT_FREQ = 0

for key, value in Y_TOKENIZER.word_counts.items():
    TOT_CNT = TOT_CNT + 1
    TOT_FREQ = TOT_FREQ + value
    if value < THRESH:
        CNT = CNT + 1
        FREQ = FREQ + value

print("% of rare words in vocabulary:", (CNT / TOT_CNT) * 100)
print("Total Coverage of rare words:", (FREQ / TOT_FREQ) * 100)

# prepare a tokenizer for reviews on training DATA
Y_TOKENIZER = Tokenizer(num_words=TOT_CNT - CNT)
Y_TOKENIZER.fit_on_texts(list(Y_TR))

# convert text sequences into integer sequences
Y_TR_SEQ = Y_TOKENIZER.texts_to_sequences(Y_TR)
Y_VAL_SEQ = Y_TOKENIZER.texts_to_sequences(Y_VAL)

# padding zero upto maximum length
Y_TR = pad_sequences(Y_TR_SEQ, maxlen=MAX_SUMMARY_LEN, padding='post')
Y_VAL = pad_sequences(Y_VAL_SEQ, maxlen=MAX_SUMMARY_LEN, padding='post')

# size of vocabulary
Y_VOC = Y_TOKENIZER.num_words + 1

Y_TOKENIZER.word_counts['sostok'], len(Y_TR)

IND = []
for i in range(len(Y_TR)):
    CNT = 0
    for j in Y_TR[i]:
        if j != 0:
            CNT = CNT + 1
    if CNT == 2:
        IND.append(i)

Y_TR = np.delete(Y_TR, IND, axis=0)
X_TR = np.delete(X_TR, IND, axis=0)

IND = []
for i in range(len(Y_VAL)):
    CNT = 0
    for j in Y_VAL[i]:
        if j != 0:
            CNT = CNT + 1
    if CNT == 2:
        IND.append(i)

Y_VAL = np.delete(Y_VAL, IND, axis=0)
X_VAL = np.delete(X_VAL, IND, axis=0)

LATENT_DIM = 300
EMBEDDING_DIM = 100
# Encoder
ENCODER_INPUTS = Input(shape=(MAX_TEXT_LEN,))

# embedding layer
ENC_EMB = Embedding(X_VOC, EMBEDDING_DIM, trainable=True)(ENCODER_INPUTS)

# encoder lstm 1
ENCODER_LSTM1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                     dropout=0.4, recurrent_dropout=0.4)
ENCODER_OUTPUT1, STATE_H1, STATE_C1 = ENCODER_LSTM1(ENC_EMB)

# encoder lstm 2
ENCODER_LSTM2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                     dropout=0.4, recurrent_dropout=0.4)
ENCODER_OUTPUT2, STATE_H2, STATE_C2 = ENCODER_LSTM2(ENCODER_OUTPUT1)

# encoder lstm 3
ENCODER_LSTM3 = LSTM(LATENT_DIM, return_state=True, return_sequences=True,
                     dropout=0.4, recurrent_dropout=0.4)
ENCODER_OUTPUTS, STATE_H, STATE_C = ENCODER_LSTM3(ENCODER_OUTPUT2)

# Set up the decoder, using `encoder_states` as initial state.
DECODER_INPUTS = Input(shape=(None,))

# embedding layer
DEC_EMB_LAYER = Embedding(Y_VOC, EMBEDDING_DIM, trainable=True)
DEC_EMB = DEC_EMB_LAYER(DECODER_INPUTS)

DECODER_LSTM = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                    dropout=0.4, recurrent_dropout=0.2)
DECODER_OUTPUTS, DECODER_FWD_STATE, DECODER_BACK_STATE = DECODER_LSTM(
    DEC_EMB,
    initial_state=[STATE_H, STATE_C])

# Attention layer
ATTN_LAYER = AttentionLayer(name='attention_layer')
ATTN_OUT, ATTN_STATES = ATTN_LAYER([ENCODER_OUTPUTS, DECODER_OUTPUTS])

# Concat attention input and decoder LSTM output
DECODER_CONCAT_INPUT = Concatenate(axis=-1, name='concat_layer')(
    [DECODER_OUTPUTS, ATTN_OUT])

# dense layer
DECODER_DENSE = TimeDistributed(Dense(Y_VOC, activation='softmax'))
DECODER_OUTPUTS = DECODER_DENSE(DECODER_CONCAT_INPUT)

# Define the model 
MODEL = Model([ENCODER_INPUTS, DECODER_INPUTS], DECODER_OUTPUTS)

MODEL.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

HISTORY = MODEL.fit([X_TR, Y_TR[:, :-1]], Y_TR.reshape(Y_TR.shape[0],
                                                       Y_TR.shape[1],
                                                       1)[:, 1:], epochs=50,
                    callbacks=[ES], batch_size=128,
                    validation_data=([X_VAL, Y_VAL[:, :-1]],
                                     Y_VAL.reshape(Y_VAL.shape[0],
                                                   Y_VAL.shape[1],
                                                   1)[:, 1:]))

REVERSE_TARGET_WORD_INDEX = Y_TOKENIZER.index_word
REVERSE_WORD_WORD_INDEX = X_TOKENIZER.index_word
TARGET_WORD_INDEX = Y_TOKENIZER.word_index

# Encode the input sequence to get the feature vector
ENCODER_MODEL = Model(inputs=ENCODER_INPUTS,
                      outputs=[ENCODER_OUTPUTS,
                               STATE_H, STATE_C])

# Decoder setup
# Below tensors will hold the states of the previous time step
DECODER_STATE_INPUT_H = Input(shape=(LATENT_DIM,))
DECODER_STATE_INPUT_C = Input(shape=(LATENT_DIM,))
DECODER_HIDDEN_STATE_INPUT = Input(shape=(MAX_TEXT_LEN,
                                          LATENT_DIM))

# Get the embeddings of the decoder sequence
DEC_EMB2 = DEC_EMB_LAYER(DECODER_INPUTS)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
DECODER_OUTPUTS2, NEW_STRING2, STATE_C2 = DECODER_LSTM(DEC_EMB2,
                                                       initial_state=[DECODER_STATE_INPUT_H,
                                                                      DECODER_STATE_INPUT_C])

# attention inference
ATTN_OUT_INF, ATTN_STATES_INF = ATTN_LAYER([DECODER_HIDDEN_STATE_INPUT,
                                            DECODER_OUTPUTS2])
DECODER_INF_CONCAT = Concatenate(axis=-1, name='concat')([DECODER_OUTPUTS2,
                                                          ATTN_OUT_INF])

# A dense softmax layer to generate prob dist. over the target vocabulary
DECODER_OUTPUTS2 = DECODER_DENSE(DECODER_INF_CONCAT)

# Final decoder model
DECODER_MODEL = Model(
    [DECODER_INPUTS] + [DECODER_HIDDEN_STATE_INPUT, DECODER_STATE_INPUT_H,
                        DECODER_STATE_INPUT_C],
    [DECODER_OUTPUTS2] + [STATE_H2, STATE_C2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = ENCODER_MODEL.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = TARGET_WORD_INDEX['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = DECODER_MODEL.predict([target_seq] +
                                                    [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = REVERSE_TARGET_WORD_INDEX[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= \
                (MAX_SUMMARY_LEN - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    new_string = ''
    for i in input_seq:
        if (i != 0 and i != TARGET_WORD_INDEX['sostok']) and \
                i != TARGET_WORD_INDEX['eostok']:
            new_string = new_string + REVERSE_TARGET_WORD_INDEX[i] + ' '
    return new_string


def seq2text(input_seq):
    new_string = ''
    for i in input_seq:
        if i != 0:
            new_string = new_string + REVERSE_WORD_WORD_INDEX[i] + ' '
    return new_string


for i in range(0, 100):
    print("Review:", seq2text(X_TR[i]))
    print("Original summary:", seq2summary(Y_TR[i]))
    print("Predicted summary:", decode_sequence(X_TR[i].reshape(1, MAX_TEXT_LEN)))
    print("\n")
