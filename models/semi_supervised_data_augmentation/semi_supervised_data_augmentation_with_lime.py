import torch
from pytorch_transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
from data_loaders.bert_classifier_data_loader import TextPreprocessing
import numpy as np
from tqdm import tqdm
from scipy.special import comb
from itertools import combinations
import random

dummy_label = 0

class TransformerExplainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = torch.load(config.saved_model_path, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        self.number_of_neighbourhood_samples = config.number_of_neighbourhood_samples
        self.number_of_features_of_explanation = config.number_of_features_of_explanation
        self.TextPreprocessor = TextPreprocessing(self.tokenizer, config.max_sentence_length)

    def predict(self, texts):
        self.model.eval()
        probabilities = []
        for text in tqdm(texts):
            ids, _ =self.TextPreprocessor.preprocess_sample(text, dummy_label)

            with torch.no_grad():
                input = torch.tensor(ids, dtype=torch.long).to(self.device)
                batch = input.reshape(1, -1)
                input = batch.transpose(0, 1).contiguous()
                logits = self.model(input,
                                    clf_tokens_mask=(input == self.tokenizer.vocab[self.TextPreprocessor.LABEL]),
                                    padding_mask=(batch == self.tokenizer.vocab[self.TextPreprocessor.PAD]))
            class_prediction = logits[0][0].cpu().numpy()
            probabilities.append([1-class_prediction, class_prediction])
        return np.array(probabilities)

    def explain(self, text):
        predictor = self.predict
        lime_explainer = LimeTextExplainer(
            split_expression=lambda x: x.split(),
            bow=False,
            class_names=["negative", "positive"]
        )
        # Make a prediction and explain it:
        exp = lime_explainer.explain_instance(
            text,
            classifier_fn=predictor,
            top_labels=2,
            num_features=self.number_of_features_of_explanation,
            num_samples=self.number_of_neighbourhood_samples,
        )
        return exp

class CreateSemiSupervisedDataSet():
    def __init__(self, config):
        self.config = config

    def intelligently_augment_sentence(self, text, label, explanation):
        model_prediction_correct = label == np.argmax(explanation.predict_proba)
        word_list = text.split()
        word_count = len(word_list)
        number_of_words_to_mask_out = round(self.config.mask_sentence_percentage * word_count)
        important_words = []
        augmented_sentences = []
        word_importance = explanation.as_list(label=label)
        for word in word_importance:
            if(word[1] >= self.config.important_word_threshold):
                important_words.append(word[0])
        number_of_important_words = len(important_words)

        number_masking_possibilities = comb(word_count - number_of_important_words, number_of_words_to_mask_out,
                                            exact=False)
        number_of_augmented_sentences = min(self.config.max_number_of_augmented_sentences_per_old_sentence,
                                            number_masking_possibilities)
        print("Generating {} sentences".format(number_of_augmented_sentences))
        print("Masking out {} words per sentence".format(number_of_words_to_mask_out))
        iteration = 0
        if(model_prediction_correct):
            # keep all the important words and randomly mask out the other words
            masking_possibilities = list(combinations(word_list, number_of_words_to_mask_out))
            random.shuffle(masking_possibilities)
            for masking_possibility in masking_possibilities:
                if(iteration >= number_of_augmented_sentences):
                    break
                important_word_in_masking_probability = False
                for important_word in important_words:
                    if important_word in masking_possibility:
                        important_word_in_masking_probability=True
                if important_word_in_masking_probability:
                    continue

                word_list_copy = word_list.copy()
                for masking_word in masking_possibility:
                    masking_word_indices = [i for i, x in enumerate(word_list_copy) if x == masking_word]
                    masking_word_index = random.choice(masking_word_indices)
                    # Todo take bette care of case when a word appears several times in the text
                    word_list_copy[masking_word_index] = "UNKWORDZ"
                augmented_sentences.append(" ".join(word_list_copy)+"\n")
                iteration+=1

        else:
            # explicitely remove the important words and randomly mast out the other words
            masking_possibilities = list(combinations(word_list, number_of_words_to_mask_out))
            random.shuffle(masking_possibilities)
            for masking_possibility in masking_possibilities:
                if (iteration >= number_of_augmented_sentences):
                    break
                word_list_copy = word_list.copy()
                number_of_current_important_words = 0
                for important_word in important_words:
                    masking_important_word_indices = [i for i, x in enumerate(word_list_copy) if x == important_word]
                    # Todo take bette care of case when a word appears several times in the text
                    for masking_important_word_index in masking_important_word_indices:
                        word_list_copy[masking_important_word_index] = "UNKWORDZ"
                        number_of_current_important_words+=1
                steps_to_do = len(masking_possibility)- number_of_current_important_words
                i = 0
                for masking_word in masking_possibility:
                    if(i>=steps_to_do):
                        break
                    if masking_word in important_words:
                        continue
                    masking_word_indices = [i for i, x in enumerate(word_list_copy) if x == masking_word]
                    masking_word_index = random.choice(masking_word_indices)
                    # Todo take bette care of case when a word appears several times in the text
                    word_list_copy[masking_word_index] = "UNKWORDZ"
                    i+=1
                augmented_sentences.append(" ".join(word_list_copy)+ "\n")
                iteration += 1
        return augmented_sentences



