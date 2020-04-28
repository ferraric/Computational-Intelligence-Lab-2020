# The sentence objects holds a sentence that we may want to embed or tag
from flair.data import Sentence, segtok_tokenizer

# Make a sentence object by passing a whitespace tokenized string
sentence = Sentence('The grass is green .')

# Print the object to see what's in there
print(sentence)

# using the token id
print(sentence.get_token(4))
# using the index itself
print(sentence[3])

for token in sentence:
    print(token)

sentence = Sentence('The grass is green.', use_tokenizer=True)

# Print the object to see what's in there
print(sentence)

for token in sentence:
    print(token)

sentence = Sentence('France is the current world cup winner.', labels=['sports', 'world cup'])

print(sentence)
for label in sentence.labels:
    print(label)