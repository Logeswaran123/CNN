# GENERATE

# import
import numpy
from keras.utils import np_utils
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# define function for generate sequence
def generate_text(model, tokenizer, max_sequence_len, seed_text, next_words):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= max_sequence_len, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text


# load cleaned text sequences
filename = 'your file name here.txt'
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
lines = text.split('\n')
seq_length = len(lines[0].split()) - 1


# load model
model = load_model('your trained model.h5')


# load tokenizer
tokenizer = load(open('your generated tokenizer.pkl', 'rb'))


# input a seed text
seed_text = " My experience at " 
# generate following new text
generated = generate_text(model, tokenizer, seq_length, seed_text, 50)
print(generated)

