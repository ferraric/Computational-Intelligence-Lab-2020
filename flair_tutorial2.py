from flair.models import SequenceTagger
from flair.data import Sentence
import time

tagger = SequenceTagger.load('ner')

sentence = Sentence('George Washington went to Washington .')

# predict NER tags
start_time = time.time()
tagger.predict(sentence)
print("prediction took:", time.time() - start_time)

# print sentence with predicted tags
print(sentence.to_tagged_string())

for entity in sentence.get_spans('ner'):
    print(entity)

quick_tagger = SequenceTagger.load('ner-fast')

start_time = time.time()
quick_tagger.predict(sentence)
print("prediction took:", time.time() - start_time)
