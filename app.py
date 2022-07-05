import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer,AutoModel,BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
from pylab import rcParams
import streamlit as st
import torchmetrics
from sklearn.preprocessing import LabelEncoder

BERT_MODEL_NAME = 'albert-base-v1'
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

class MeshNetwork(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.bert = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=13,return_dict=True)
    self.criterion = F.cross_entropy

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    return output.logits
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    y = batch['labels']
    y_hat = self.forward(input_ids, attention_mask)
    loss = self.criterion(y_hat, y)
    # Calculate acc
    predictions = F.softmax(y_hat, dim=1).argmax(dim=1)
    acc = torchmetrics.functional.accuracy(predictions, y)
    self.log("train_acc", acc, on_step=False,prog_bar=True, on_epoch=True, logger=True)
    self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
    return {"loss": loss, "predictions": y_hat, "labels": y}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    y = batch["labels"]
    y_hat = self.forward(input_ids, attention_mask)
    loss = self.criterion(y_hat, y)    
    predictions = F.softmax(y_hat, dim=1).argmax(dim=1)
    acc = torchmetrics.functional.accuracy(predictions, y)
    self.log("val_acc", acc, prog_bar=True, on_step = False,on_epoch=True, logger=True)
    self.log("val_loss", loss, prog_bar=True, on_epoch = True, logger=True)

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    y = batch["labels"]
    y_hat = self.forward(input_ids, attention_mask)
    loss = self.criterion(y_hat, y)    
    predictions = F.softmax(y_hat, dim=1).argmax(dim=1)
    acc = torchmetrics.functional.accuracy(predictions, y)
    self.log("test_acc", acc, prog_bar=True, on_step=False,on_epoch=True, logger=True)
    self.log("test_loss", loss, prog_bar=True, on_epoch = True, logger=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(params = self.parameters())
    return optimizer




with st.spinner("Loading model..."):
    model = torch.load("complete.pt")

st.success("Model loaded.")

if st.button("predict"):
    st.write("predicted")
