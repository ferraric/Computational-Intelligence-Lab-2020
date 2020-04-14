from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler, PiecewiseLinear, create_lr_scheduler_with_warmup, ProgressBar
import torch
import numpy as np
import pandas as pd


class BertClassifierTrainer():
    def __init__(self, config, device, optimizer, model, tokenizer, TextProcessor,
                 train_dataset, validation_dataset, test_dataset):
        self.config = config
        self.device = device
        self.optimizer = optimizer
        self.model = model
        self.tokenizer = tokenizer
        self.TextProcessor = TextProcessor
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.test_predictions = None

    def learn(self, engine, batch):
        self.model.train()
        inputs, labels = (t.to(self.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # [S, B]
        _, loss = self.model(inputs,
                        clf_tokens_mask=(inputs == self.tokenizer.vocab[self.TextProcessor.LABEL]),
                        clf_labels=labels)
        #loss = loss / self.config.gradient_acc_steps
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
        #if engine.state.iteration % self.config.gradient_acc_steps == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


    def validation_inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch, labels = (t.to(self.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()
            logits = self.model(inputs,
                           clf_tokens_mask = (inputs == self.tokenizer.vocab[self.TextProcessor.LABEL]),
                           padding_mask = (batch == self.tokenizer.vocab[self.TextProcessor.PAD]))
        return logits, labels

    def test_inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch, labels = (t.to(self.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()
            logits = self.model(inputs,
                                clf_tokens_mask=(inputs == self.tokenizer.vocab[self.TextProcessor.LABEL]),
                                padding_mask=(batch == self.tokenizer.vocab[self.TextProcessor.PAD]))
        # we will now save the predicted labels to an output file.
        predictions = torch.argmax(logits, axis=1).cpu().numpy()
        return predictions

    def on_test_iteration_completed(self, engine):
        predictions = np.where(engine.state.output == 0, -1, engine.state.output)
        if(self.test_predictions is None):
            self.test_predictions = predictions
        else:
            self.test_predictions = np.concatenate((self.test_predictions, predictions))
            print(self.test_predictions)
            return

    def train_n_validate(self):
        trainer = Engine(self.learn)
        evaluator = Engine(self.validation_inference)
        Accuracy().attach(evaluator, "validation_accuracy")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.validation_dataset)
            print(f"validation epoch: {engine.state.epoch} acc: {100 * evaluator.state.metrics['validation_accuracy']}")

        scheduler = PiecewiseLinear(self.optimizer,
                                    'lr', [(0, 0.0), (self.config.n_warmup, self.config.learning_rate),
                                                      (len(self.train_dataset) * self.config.num_epochs, 0.0)])
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        # add progressbar with loss
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
        ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

        # save checkpoints and finetuning config
        checkpoint_handler = ModelCheckpoint(self.config.summary_dir, 'config_checkpoint',
                                             require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'bert_classifier_model': self.model})
        trainer.run(self.train_dataset, max_epochs=self.config.num_epochs)

    def test(self):
        test_evaluator = Engine(self.test_inference)
        test_evaluator.add_event_handler(Events.ITERATION_COMPLETED, self.on_test_iteration_completed)
        test_evaluator.run(self.test_dataset)
        id = np.arange(1,self.test_predictions.shape[0]+1)
        prediction_dataframe = pd.DataFrame(id, columns=["Id"])
        prediction_dataframe["Prediction"] = self.test_predictions
        prediction_dataframe.to_csv(self.config.test_prediction_path)