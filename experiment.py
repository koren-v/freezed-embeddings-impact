from typing import Dict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from optimization import RAdam, Lookahead
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

        for experiment_name, train_loader in dataloaders_dicts.items():

            for freeze_embedding in True, False:

                model = AutoModelForSequenceClassification.from_pretrained(
                    self._pretrained_model, config=config
                )

                model.base_model.embeddings.requires_grad_(not freeze_embedding)

                base_optimizer = RAdam(
                    [
                        {"params": model.base_model.parameters(), "lr": self._base_model_lr},
                        {"params": model.classifier.parameters(), "lr": self._classifier_lr}
                    ]
                )

                optimizer = Lookahead(base_optimizer)

                num_training_steps = self._num_epochs * len(train_loader) / self._grad_accumulation_step
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                )

                comment = f"{experiment_name}_freeze_embeddings_{freeze_embedding}".lower()
                logger = SummaryWriter(self._log_path, comment=comment)

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
                    num_epochs=self._num_epochs,
                    dataloaders_dict=dict(train=train_loader, valid=self._validation_loader)
                )

