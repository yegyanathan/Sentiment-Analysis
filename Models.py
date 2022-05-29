import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI


class RottenTomatoesClassifier(pl.LightningModule):

    """
    Bert model for sentiment classification.
    Data: Rotten Tomatoes movie reviews.
    
    """

    def __init__(self,
            model_name = 'bert-base-uncased',
            debug = False,
            pretrained = False,
            lr = 3e-4,
            momentum = 0.9,
            epochs = 2,
            weight_decay = 1e-5,
            batch_size = 16,
            seq_length = 28,
            percent = 4):

        super(RottenTomatoesClassifier, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.percent = percent
        self.seq_length = seq_length
        self.epochs = epochs
        self.debug = debug

        self.train_ds = None
        self.validation_ds = None
        self.test_ds = None

        self.model = transformers.BertForSequenceClassification.from_pretrained(self.model_name, return_dict = False)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        self.loss = torch.nn.CrossEntropyLoss(reduction = 'none')


    def prepare_data(self):

        """
        Download dataset, tokenize etc.
        split, transforms

        """

        def _tokenize(x):

            x['input_ids'] = self.tokenizer.batch_encode_plus(
                    x['text'], 
                    add_special_tokens = True,
                    max_length = 32,
                    pad_to_max_length = True,)['input_ids']

            return x

        def _prepare_ds(split):

            ds = load_dataset('rotten_tomatoes', split = split)
            ds = ds.map(_tokenize, batched  =True)
            ds.set_format(type = 'torch', columns = ['input_ids', 'label'])
            #ds.shuffle(seed = 42)

            return ds

        train_ds, validation_ds, test_ds = map(_prepare_ds, ('train', 'validation', 'test'))
        self.train_ds = train_ds.shuffle(seed = 42)
        self.validation_ds = validation_ds.shuffle(seed = 42)
        self.test_ds = test_ds.shuffle(seed = 42)



    def forward(self, input_ids):

        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)

        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()

        self.log(name = 'train_loss', value = loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):

        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):

        loss = torch.cat([o['val_loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['val_acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        self.log(name = 'val_loss', value = loss)
        self.log(name = 'val_acc', value = acc)
        return {**out, 'log': out}


    def train_dataloader(self):

        return DataLoader(
                self.train_ds,
                batch_size = self.batch_size,
                drop_last = True,
                shuffle = True,)

    def val_dataloader(self):

        return DataLoader(
                self.validation_ds,
                batch_size = self.batch_size,
                drop_last = True,
                shuffle = True,)
    """

    def test_dataloader(self):
    
    return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            drop_last = True,
            shuffle = True,)
    
    
    """
    
    def configure_optimizers(self):

        return torch.optim.SGD(
            self.model.parameters(),
            lr = self.lr,
            momentum = self.momentum)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length = 120,
                return_token_type_ids = False,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt',
                truncation = True,
        )

        y_hat = self.model(encoding['input_ids'])
        return y_hat.argmax().item()


def main():

    model = RottenTomatoesClassifier()

    checkpoint_callback = ModelCheckpoint(dirpath = 'my_model',
                                        every_n_epochs = 1,
                                        monitor = 'val_loss',
                                        mode = 'min')

    logger = TensorBoardLogger('logs/', name = 'rotten-tomatoes', version=0)
    

    trainer = Trainer(
        default_root_dir = 'logs',
        gpus = (1 if torch.cuda.is_available() else 0),
        callbacks = [checkpoint_callback],
        max_epochs = model.epochs,
        logger = logger)

    trainer.fit(model)


if __name__ == '__main__':

    main()