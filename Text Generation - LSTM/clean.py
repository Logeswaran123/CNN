# LSTM - CLEAN TEXT FILE

# import
import string

# save tokens to file
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


# load document
filename = 'your prepared file here.txt'
file = open(filename, 'r')
# read all text
doc = file.read()
# close the file
file.close()
print(doc[:200])


# clean the document
# replace '-' & '--' with a space ' '
doc = doc.replace('--', ' ')
doc = doc.replace('-', ' ')
# split into tokens by white space
tokens = doc.split()
# remove punctuation from each token
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]
# remove non alphabetic tokens
tokens = [word for word in tokens if word.isalpha()]
# make all as lower case
tokens = [word.lower() for word in tokens]
print('TOTAL TOKENS: %d' % len(tokens))
print('UNIQUE TOKENS: %d' % len(set(tokens)))


# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store the line
	sequences.append(line)
print('TOTAL SEQUENCES: %d' % len(sequences))


# save sequences to file
output_file = 'cleaned.txt'
save_doc(sequences, output_file)
