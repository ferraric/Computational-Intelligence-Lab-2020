from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter,TextClassifierParamSelector, OptimizationValue

flair_data_path = 'data/transformed_data'

# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label",}
# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus = CSVClassificationCorpus(flair_data_path,
                                 column_name_map,
                                 skip_header=True,
                                 delimiter=',',).downsample(0.01)

print(corpus)

search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ WordEmbeddings('glove') ],
    [ FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') ],
    [ WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') ],
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[128, 256, 512])
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

# create the parameter selector
param_selector = TextClassifierParamSelector(
    corpus,
    False,
    'param_tuning',
    'lstm',
    max_epochs=100,
    training_runs=3,
    optimization_value=OptimizationValue.DEV_SCORE,
    anneal_factor=0.5,
    patience=5,
)

# start the optimization
param_selector.optimize(search_space, max_evals=100)