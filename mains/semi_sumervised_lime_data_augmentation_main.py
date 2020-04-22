import os, sys, inspect, random, json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utilities.general_utilities import get_args
from utilities.config import process_config
from utilities.parse_data_files import parse_train_files
from models.semi_supervised_data_augmentation.semi_supervised_data_augmentation_with_lime import TransformerExplainer, CreateSemiSupervisedDataSet
import torch


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    with open(
            os.path.join(config.output_folder, "config_summary.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataframe = parse_train_files(config.train_dataset_path, True, False)
    Explainer = TransformerExplainer(config, device)
    DatasetCreator = CreateSemiSupervisedDataSet(config)
    positive_sentences_file = open(os.path.join(config.output_folder, "train_pos_augmented.txt"), "w+")
    negative_sentences_file = open(os.path.join(config.output_folder, "train_neg_augmented.txt"), "w+")
    number_of_sentences = train_dataframe.shape[0]
    iter = 0
    for index, row in train_dataframe.iterrows():
        print("Sentence {} of {}".format(iter, number_of_sentences))
        text = row["text"]
        label = row["label"]
        explanation = Explainer.explain(text)
        if(config.randomly_augment_sentences):
            pass
        else:
            augmented_sentences = DatasetCreator.intelligently_augment_sentence(text, label, explanation)
            if(label==1):
                positive_sentences_file.writelines(augmented_sentences)
            else:
                negative_sentences_file.writelines(augmented_sentences)
        iter+=1



if __name__ == '__main__':
   main()