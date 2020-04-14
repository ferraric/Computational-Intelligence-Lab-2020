import os, sys, inspect, random, json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utilities.general_utilities import get_args, create_dirs
from utilities.config import process_config
from comet_ml import Experiment
from utilities.parse_data_files import parse_train_files, parse_test_file, remove_line_index_and_comma_of_sentence
from pytorch_transformers import BertTokenizer
from data_loaders.bert_classifier_data_loader import TextPreprocessing, BertClassifierDataLoader
from models.bert_classifier_model import TransformerWithClfHead
from trainers.bert_classifier_trainer import BertClassifierTrainer
from pytorch_transformers import cached_path
import torch
from pytorch_transformers.optimization import AdamW





def setup_experiment(experiment, config):
    if config.use_comet_experiments:
        experiment_id = experiment.connection.experiment_id
    else:
        experiment_id = str(random.randint(1, 1000000))

    config.summary_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "summary/"
    )
    config.checkpoint_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "checkpoint/"
    )
    create_dirs([config.summary_dir, config.checkpoint_dir])
    print("...creating folder {}".format(config.summary_dir))

    with open(
            os.path.join(config.summary_dir, "config_summary.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    setup_experiment(experiment, config)
    experiment.log_asset(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader and Model
    ############################
    train_dataframe = parse_train_files(config.train_dataset_path, True, False)
    test_dataframe = parse_test_file(config.test_dataset_path, remove_line_index_and_comma_of_sentence)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    TextProcessor = TextPreprocessing(tokenizer, config.max_sentence_length)
    DataLoader = BertClassifierDataLoader()

    train_dataset, validation_dataset = DataLoader.create_dataloader(train_dataframe, TextProcessor,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           validation_percentage=config.validation_percentage)

    test_dataset = DataLoader.create_dataloader(test_dataframe, TextProcessor,
                                batch_size=config.batch_size,
                                shuffle=False,
                                validation_percentage=None)

    # download pre-trained model and config
    state_dict = torch.load(cached_path("https://s3.amazonaws.com/models.huggingface.co/"
                                        "naacl-2019-tutorial/model_checkpoint.pth"), map_location='cpu')

    bert_config = torch.load(cached_path("https://s3.amazonaws.com/models.huggingface.co/"
                                    "naacl-2019-tutorial/model_training_args.bin"))

    # init model: Transformer base + classifier head
    model = TransformerWithClfHead(bert_config=bert_config, fine_tuning_config=config).to(device)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Parameters discarded from the pretrained model: {incompatible_keys.unexpected_keys}")
    print(f"Parameters added in the model: {incompatible_keys.missing_keys}")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    Trainer = BertClassifierTrainer(config, device, optimizer, model, tokenizer, TextProcessor, train_dataset,
                validation_dataset, test_dataset, experiment)
    Trainer.train_n_validate()
    Trainer.test()

if __name__ == '__main__':
    main()