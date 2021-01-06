from typing import Dict
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from optimization import RAdam
from learner import Learner


class Experiment:
    def __init__(
            self,
            validation_loader,
            metric,
            loss_function,
            num_epochs,
            num_classes,
            log_path,
            pretrained_model="xlm-roberta-base",
            grad_accumulation_step=1,
            clip_grad_norm=1.0,
            base_model_learning_rate=1e-5,
            classifier_learning_rate=1e-3,
    ):
        self._validation_loader = validation_loader
        self._metric = metric
        self._loss_function = loss_function
        self._num_epochs = num_epochs
        self._num_classes = num_classes
        self._pretrained_model = pretrained_model
        self._grad_accumulation_step = grad_accumulation_step
        self._clip_grad_norm = clip_grad_norm
        self._base_model_lr = base_model_learning_rate
        self._classifier_lr = classifier_learning_rate
        self._log_path = log_path

    def run(self, dataloaders_dicts: Dict[str, DataLoader]):
        config = AutoConfig.from_pretrained(self._pretrained_model)
        config.num_labels = self._num_classes

        max_train_steps = self._get_max_steps(
            dataloaders_dicts, self._num_epochs, self._grad_accumulation_step
        )

        for experiment_name, train_loader in dataloaders_dicts.items():

            for freeze_embedding in True, False:

                model = AutoModelForSequenceClassification.from_pretrained(
                    self._pretrained_model, config=config
                )

                model.base_model.embeddings.requires_grad_(not freeze_embedding)

                optimizer = RAdam(
                    [
                        {"params": model.base_model.parameters(), "lr": self._base_model_lr},
                        {"params": model.classifier.parameters(), "lr": self._classifier_lr}
                    ]
                )

                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=0, num_training_steps=max_train_steps
                )

                comment = f"{experiment_name}_freeze_embeddings_{freeze_embedding}".lower()
                experiment_path = self._log_path + "/" + comment
                if not os.path.exists(experiment_path):
                    os.mkdir(experiment_path)
                logger = SummaryWriter(experiment_path)

                learner = Learner(
                    model=model,
                    optimizer=optimizer,
                    loss_function=self._loss_function,
                    metric=self._metric,
                    logger=logger,
                    scheduler=scheduler,
                    grad_accumulation_step=self._grad_accumulation_step,
                    clip_grad_norm=self._clip_grad_norm
                )

                learner.fit(
                    dataloaders_dict=dict(train=train_loader, valid=self._validation_loader),
                    max_steps=max_train_steps,
                    num_epochs=None,
                )

    @staticmethod
    def _get_max_steps(train_loaders_dict, num_epochs, grad_accumulation_step):
        max_epoch_steps = max(len(loader) for loader in train_loaders_dict.values())
        return int(num_epochs * max_epoch_steps / grad_accumulation_step)
