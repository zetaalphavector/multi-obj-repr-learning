import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import Adafactor, get_linear_schedule_with_warmup

from losses import MultipleNegativesRankingLoss, TripletLoss
from utils.datasets import get_collate_fn, get_sampler, get_train_dev_datasets


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Specter(pl.LightningModule):
    def __init__(self, init_args):
        super().__init__()
        self.save_hyperparameters(init_args)
        print(self.hparams)
        self.model = AutoModel.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nreimers/MiniLM-L6-H384-uncased"
        )

        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.hparams.seqlen = self.model.config.max_position_embeddings

        try:
            if self.hparams.loss == "triplet":
                self.objective = TripletLoss()
            elif self.hparams.loss == "mnrl":
                self.objective = MultipleNegativesRankingLoss()
            else:
                assert False, f"loss: {self.hparams.loss} not supported"
        except AttributeError:
            self.objective = MultipleNegativesRankingLoss()

        # This is a dictionary to save the embeddings for source papers in test step.
        self.embedding_output = {}

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if self.hparams.pooling == "mean":
            embedding = mean_pooling(output, attention_mask)
        elif self.hparams.pooling == "cls":
            embedding = output[0][:, 0, :]
        elif self.hparams.pooling == "pretrain":
            embedding = output[1]
        else:
            assert False, f"{self.hparams.pooling} not supported"
        return embedding

    def setup(self, stage=None):
        self.train_dataset, self.dev_datasets = get_train_dev_datasets(
            data_dirs=self.hparams.data_dirs
        )

    def train_dataloader(self):
        collate_fn = get_collate_fn(
            self.tokenizer,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=512,
        )
        sampler = get_sampler(
            dataset=self.train_dataset,
            sampling=self.hparams.sampling_method,
            batch_size=self.hparams.batch_size,
            dataset_weights=self.hparams.dataset_weights,
        )
        return DataLoader(
            self.train_dataset,
            num_workers=self.hparams.num_workers,
            batch_sampler=sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        collate_fn = get_collate_fn(
            self.tokenizer,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=512,
        )
        val_dataloaders = []
        self.idx_to_val_dataset = {}
        for idx, dev_dataset in enumerate(self.dev_datasets):
            if dev_dataset.name in [
                name for name, _ in self.idx_to_val_dataset.values()
            ]:
                continue
            val_dataloader = DataLoader(
                dev_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            val_dataloaders.append(val_dataloader)
            self.idx_to_val_dataset[idx] = (dev_dataset.name, len(dev_dataset))

        return val_dataloaders

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        if self.hparams.steps is None:
            num_devices = 1  # only 1 GPU support
            effective_batch_size = (
                self.hparams.batch_size * self.hparams.grad_accum * num_devices
            )
            dataset_size = len(self.train_dataset)
            return (dataset_size / effective_batch_size) * self.hparams.epochs
        else:
            return self.hparams.steps

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.lr,
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.lr,
                eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        source_embedding = self.forward(**batch[0])
        pos_embedding = self.forward(**batch[1])
        neg_embedding = self.forward(**batch[2])

        loss = self.objective(source_embedding, pos_embedding, neg_embedding)
        lr_scheduler = self.lr_schedulers()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "lr",
            lr_scheduler.get_last_lr()[-1],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        source_embedding = self.forward(**batch[0])
        pos_embedding = self.forward(**batch[1])
        neg_embedding = self.forward(**batch[2])
        loss = self.objective(source_embedding, pos_embedding, neg_embedding)
        dataset_name = self.idx_to_val_dataset[dataloader_idx][0]
        self.log(
            f"val_loss/{dataset_name}",
            loss,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
        )
        # Manually accumulate outputs for aggregation.
        if not hasattr(self, "_val_outputs"):
            self._val_outputs = {}
        self._val_outputs.setdefault(dataloader_idx, []).append(loss.detach())

    def _compute_avg_val_losses(self) -> dict:
        results = {}
        for idx, (name, size) in self.idx_to_val_dataset.items():
            losses = self._val_outputs.get(idx, [])
            if losses:
                avg_loss = torch.stack(losses).mean()
                results[name] = avg_loss.item()
                self.logger.experiment.add_scalars(
                    "avg_val_loss", {name: results[name]}, self.global_step
                )
        total = sum(
            size
            for _, (name, size) in self.idx_to_val_dataset.items()
            if name in results
        )
        if total > 0:
            weighted = sum(
                results[name] * size
                for _, (name, size) in self.idx_to_val_dataset.items()
                if name in results
            )
            results["mean"] = weighted / total
        else:
            results["mean"] = None
        return results

    def on_validation_epoch_end(self) -> None:
        avg_val_losses = self._compute_avg_val_losses()
        if avg_val_losses.get("mean") is not None:
            self.log(
                "avg_val_loss", avg_val_losses["mean"], on_epoch=True, prog_bar=True
            )
            self.logger.experiment.add_scalars(
                "avg_val_loss", {"mean": avg_val_losses["mean"]}, self.global_step
            )
        self._val_outputs = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="path to the model (if not setting checkpoint)",
    )
    parser.add_argument(
        "--data_dirs",
        help="Space-separated list of directory paths containing the datasets.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--sampling",
        dest="sampling_method",
        default="mixed",
        type=str,
        help="Sampling method to use (mixed, batch)",
    )
    parser.add_argument(
        "--weights",
        dest="dataset_weights",
        help="Space-separated list of integers mapping one-to-one to the list of data_dirs.",
        nargs="+",
        type=int,
        required=False,
    )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--loss", default="mnrl", type=str, help="Loss to use (mnrl, triplet)"
    )
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--limit_test_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--fast_dev_run", default=None, type=int)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
    )
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument(
        "--pooling",
        default="mean",
        type=str,
        help="pooling mechanism during training (mean, cls, pretrain).",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--version", help="pytorch lightning sub-directory name for output"
    )
    args = parser.parse_args()
    print(args)
    return args


def get_train_params(args):
    train_params = {}
    train_params["precision"] = 16 if args.fp16 else 32
    train_params["accumulate_grad_batches"] = args.grad_accum
    train_params["limit_val_batches"] = args.limit_val_batches

    if (
        args.val_check_interval > 1.0
    ):  # val_check_interval corresponds to the training steps
        train_params["val_check_interval"] = int(args.val_check_interval)
    else:  # val_check_interval corresponds training set proportion (doesn't work for IterableDataset)
        train_params["val_check_interval"] = args.val_check_interval

    train_params["max_steps"] = -1 if args.steps is None else args.steps
    train_params["max_epochs"] = args.epochs
    train_params["log_every_n_steps"] = 1

    if torch.cuda.is_available():
        train_params["accelerator"] = "gpu"
        train_params["devices"] = args.gpus
    else:
        train_params["accelerator"] = "cpu"
        train_params["devices"] = 1

    if args.fast_dev_run:
        train_params["fast_dev_run"] = args.fast_dev_run
    return train_params


def main():
    args = parse_args()

    pl.seed_everything(seed=args.seed, workers=True)

    if args.num_workers == 0:
        print("num_workers cannot be less than 1")
        return

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.checkpoint_path is not None:
        print("Loading from checkpoint")
        model = Specter.load_from_checkpoint(args.checkpoint_path, init_args=vars(args))
    else:
        print("Training from scratch")
        model = Specter(args)

    logger = TensorBoardLogger(save_dir=args.save_dir, version=args.version)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints/",
        filename="{epoch}_{step}_{avg_val_loss:.3f}",
        save_top_k=1,
        verbose=True,
        monitor="avg_val_loss",  # monitors metrics logged by self.log.
        mode="min",
    )

    progress_callback = TQDMProgressBar(refresh_rate=50)

    early_stopping_callback = EarlyStopping(
        monitor="avg_val_loss", patience=2, mode="min", verbose=True
    )

    extra_train_params = get_train_params(args)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            progress_callback,
            early_stopping_callback,
        ],
        enable_checkpointing=True,
        **extra_train_params,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
