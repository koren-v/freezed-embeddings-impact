from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_


class Learner:
    def __init__(self, model, optimizer, loss_function, metric,
                 logger, scheduler=None, grad_accumulation_step=1,
                 log_grads_every=10, **kwargs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = model.to(device)
        self._metric = metric
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._grad_accumulation_step = grad_accumulation_step
        self._logger = logger
        self._log_grads_every = log_grads_every

        self.kwargs = kwargs
        self._steps = 0

    def fit(self, dataloaders_dict, *, num_epochs=None, max_steps=None):

        phases = "train", "valid"

        assert (num_epochs is None) != (max_steps is None), \
            "You must specify num_epochs or max_steps argument"

        if num_epochs is None:
            num_epochs = max_steps // len(dataloaders_dict["train"]) + 1

        for epoch in tqdm(range(1, num_epochs + 1)):

            for phase in phases:

                loader = dataloaders_dict[phase]
                epoch_dict = self.epoch(phase=phase, dataloader=loader, max_steps=max_steps)
                if epoch_dict is None:
                    break

                self._logger.add_scalar("{} Epoch Loss".format(phase.title()), epoch_dict["epoch_loss"], epoch)
                self._logger.add_scalar("{} Epoch Metric".format(phase.title()), epoch_dict["epoch_metric"], epoch)

        self._logger.close()

    def epoch(self, phase, dataloader, max_steps=None):

        self._model.train() if phase == "train" else self._model.eval()

        running_loss = 0.0
        num_examples = 0

        epoch_logits = []
        epoch_labels = []

        for step, batch in enumerate(dataloader):

            if phase == "train":
                self._steps += 1
                if self._steps > max_steps:
                    return

            batch = self._batch_to_device(batch)
            labels = batch[-1]

            with torch.set_grad_enabled(phase == "train"):

                outputs = self._forward_step(batch)
                loss = self._compute_loss(outputs=outputs, labels=labels)

                loss /= self._grad_accumulation_step

                if phase == "train":
                    # loss.backward()
                    loss.backward()

                    # clip_grad_norm
                    if self.kwargs.get("clip_grad_norm") is not None:
                        clip_grad_norm_(self._model.parameters(), self.kwargs["clip_grad_norm"])

                    # log grads
                    if (step + 1) % self._log_grads_every == 0:
                        self._log_layer_grads(step=step)

                    # optimizer.step()
                    if (step + 1) % self._grad_accumulation_step == 0:
                        self._optimizer.step()
                        if self._scheduler is not None:
                            self._scheduler.step()
                        self._optimizer.zero_grad()

            epoch_logits.extend(outputs.detach().cpu().tolist())
            epoch_labels.extend(labels.detach().cpu().tolist())

            # statistics
            running_loss += loss.item() * labels.size(0)
            num_examples += labels.size(0)

        epoch_loss = running_loss / num_examples
        epoch_metric = self._metric(epoch_labels, epoch_logits)

        epoch_dict = {
            "epoch_loss": epoch_loss,
            "epoch_metric": epoch_metric,
            "epoch_logits": epoch_logits,
            "epoch_labels": epoch_labels
        }

        return epoch_dict

    def _batch_to_device(self, batch):
        return [tensor.to(self._device) for tensor in batch]

    def _forward_step(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        return self._model(input_ids=input_ids, attention_mask=attention_mask)[0]

    def _compute_loss(self, outputs, labels):
        loss = self._loss_function(outputs, labels.long())
        return loss

    def _log_layer_grads(self, step):
        # embeddings
        embeddings_grads = torch.tensor([], device=self._device)
        for parameter in self._model.base_model.embeddings.parameters():
            if parameter.grad is not None:
                flattened_grads = parameter.grad.clone().reshape(-1)
                embeddings_grads = torch.cat((embeddings_grads, flattened_grads))
        if len(embeddings_grads) > 0:
            self._logger.add_histogram("Embeddings Gradients", embeddings_grads, step)

        # layers
        for i, layer in enumerate(self._model.base_model.encoder.layer):
            layer_grads = torch.tensor([], device=self._device)
            for parameter in layer.parameters():
                if parameter.grad is not None:
                    flattened_grads = parameter.grad.clone().reshape(-1)
                    layer_grads = torch.cat((layer_grads, flattened_grads))
            if len(layer_grads) > 0:
                self._logger.add_histogram(f"Layer {i+1} Gradients", layer_grads, step)

        # classifier
        classifier_grads = torch.tensor([], device=self._device)
        for parameter in self._model.classifier.parameters():
            if parameter.grad is not None:
                flattened_grads = parameter.grad.clone().reshape(-1)
                classifier_grads = torch.cat((classifier_grads, flattened_grads))
        if len(classifier_grads) > 0:
            self._logger.add_histogram("Classifier Gradients", classifier_grads, step)
