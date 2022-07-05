import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer,AutoModel,BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import streamlit as st
import torchmetrics
pwd = os.path.dirname(__file__)
MODEL_PATH = os.path.join(pwd,"data.pt")
print(MODEL_PATH)

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



st.title("MeSH Classify")

with st.spinner("Loading model..."):
    model = torch.load(MODEL_PATH)

st.success("Model loaded.")
user_input = st.text_input("Enter text to be classified.")
st.write("Check MeSH categories: [link](https://www.ncbi.nlm.nih.gov/mesh/1000048)")
st.markdown("***")


if st.button("Classify Text"):
    if user_input:
      encoding = tokenizer.encode_plus(
      user_input,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
      )
      input_ids=encoding["input_ids"].flatten()
      attention_mask=encoding["attention_mask"].flatten()


      y_hat = model(input_ids=input_ids.reshape(-1, 512), 
        attention_mask = attention_mask.reshape(-1, 512))
      prob = F.softmax(y_hat, dim=1)
      st.write(prob)
      predictions = prob.argmax(dim=1)
      st.write(predictions)


