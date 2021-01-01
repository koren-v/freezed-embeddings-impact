from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_


class Learner:
    def __init__(self, model, optimizer, loss_function, metric,
                 logger, scheduler=None, grad_accumulation_step=1, **kwargs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.metric = metric
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_accumulation_step = grad_accumulation_step
        self.logger = logger

        self.kwargs = kwargs

    def fit(self, num_epochs, dataloaders_dict):

        phases = ["train", "valid"]

        for epoch in range(1, num_epochs + 1):

            for phase in phases:

                loader = dataloaders_dict[phase]
                epoch_dict = self.epoch(phase=phase,
                                        dataloader=loader,
                                        current_epoch=epoch,
                                        num_epochs=num_epochs)

                self.logger.add_scalar("{} Epoch Loss".format(phase.title()), epoch_dict["epoch_loss"], epoch)
                self.logger.add_scalar("{} Epoch Metric".format(phase.title()), epoch_dict["epoch_metric"], epoch)

    def epoch(self, phase, dataloader, current_epoch, num_epochs):

        self.model.train() if phase == "train" else self.model.eval()

        running_loss = 0.0
        num_examples = 0

        epoch_logits = []
        epoch_labels = []

        with tqdm(dataloader) as progress:

            for step, batch in enumerate(dataloader):

                batch = self._batch_to_device(batch)
                labels = batch[-1]

                with torch.set_grad_enabled(phase == "train"):

                    outputs = self._forward_step(batch)
                    loss = self._compute_loss(outputs=outputs, labels=labels)

                    loss /= self.grad_accumulation_step

                    if phase == "train":
                        # loss.backward()
                        loss.backward()

                        # clip_grad_norm
                        if self.kwargs.get("clip_grad_norm") is not None:
                            clip_grad_norm_(self.model.parameters(), self.kwargs["clip_grad_norm"])

                        # optimizer.step()
                        if (step + 1) % self.grad_accumulation_step == 0:
                            self.optimizer.step()
                            if self.scheduler is not None:
                                self.scheduler.step()
                            self.optimizer.zero_grad()
                            self._log_layer_grads(step=step)

                # collecting raw outputs to find top losses examples
                epoch_logits.extend(outputs.detach().cpu().tolist())
                epoch_labels.extend(labels.detach().cpu().tolist())

                # statistics
                running_loss += loss.item() * labels.size(0)
                num_examples += labels.size(0)

                self._print_batch_statistics(bar=progress,
                                             current_epoch=current_epoch,
                                             loss=loss,
                                             num_epochs=num_epochs)

            epoch_loss = running_loss / num_examples
            epoch_metric = self.metric(epoch_labels, epoch_logits)

            epoch_dict = {
                "epoch_loss": epoch_loss,
                "epoch_metric": epoch_metric,
                "epoch_logits": epoch_logits,
                "epoch_labels": epoch_labels
            }

            self._print_epoch_statistics(bar=progress,
                                         epoch=current_epoch,
                                         phase=phase,
                                         epoch_dict=epoch_dict,
                                         num_epochs=num_epochs)

        return epoch_dict

    def _batch_to_device(self, batch):
        return [tensor.to(self.device) for tensor in batch]

    def _forward_step(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

    def _compute_loss(self, outputs, labels):
        loss = self.loss_function(outputs, labels.long())
        return loss

    def _log_layer_grads(self, step):
        for name, weight in self.model.named_parameters():
            if weight.grad is not None:
                self.logger.add_histogram(f"{name}.grad", weight.grad, step)

    def _print_batch_statistics(self, bar, current_epoch, loss, num_epochs):
        stats = "Epoch [{}/{}], Loss: {:.4f}".format(
            current_epoch, num_epochs, loss.item() * self.grad_accumulation_step
        )
        bar.set_description(stats)
        bar.update()

    @staticmethod
    def _print_epoch_statistics(bar, epoch, phase, epoch_dict, num_epochs):
        stats = "Epoch [{}/{}],  {} Loss:  {:.4f}  {} Metric:  {:.4f}  |".format(
            epoch, num_epochs,
            phase.title(), epoch_dict["epoch_loss"],
            phase.title(), epoch_dict["epoch_metric"]
        )
        bar.set_description(stats)
