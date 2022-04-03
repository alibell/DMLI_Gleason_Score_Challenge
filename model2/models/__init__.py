from torchvision.models import efficientnet_b0
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import openslide
import hashlib
import random
import torch
import os
import pickle
from torchvision import transforms
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from PIL import Image


class patchClassifier (nn.Module):
  def __init__ (self):
    super().__init__()

    self.backbone = efficientnet_b0(pretrained=True)
    classifier_input = self.backbone.classifier[1].in_features
    self.backbone.classifier[1] = nn.Sequential(
        nn.Linear(classifier_input, 4)
    )

    for parameter in self.backbone.parameters():
      parameter.requires_grad = True

    self.network = self.backbone

    self.criterion = nn.CrossEntropyLoss()
    self.optim = optim.AdamW(self.parameters())

  def forward(self, x):
    predictions = self.network(x)
    y_hat = predictions

    return y_hat

  def predict(self, x):
    self.eval()

    with torch.no_grad():
      y_hat = self.forward(x)

      return y_hat

  def fit(self, x, y):
    self.train()
    self.optim.zero_grad()
    
    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    loss.backward()

    self.optim.step()

    loss_ = loss.detach().cpu().item()

    return loss_

class tilesClassifier (nn.Module):
  def __init__ (self):
    super().__init__()

    self.backbone = efficientnet_b0(pretrained=True)
    classifier_input = self.backbone.classifier[1].in_features
    self.backbone.classifier[1] = nn.Sequential(
        nn.Linear(classifier_input, 6)
    )

    for parameter in self.backbone.parameters():
      parameter.requires_grad = True

    self.softmax = nn.Softmax(dim=0)

    self.network = self.backbone

    self.criterion = nn.CrossEntropyLoss()
    self.optim = optim.AdamW(self.parameters())

  def forward(self, x):
    predictions = self.network(x)
    y_hat = predictions

    return y_hat

  def predict(self, x):
    self.eval()

    with torch.no_grad():
      y_hat = self.forward(x)
      y_hat = self.softmax(y_hat)

      return y_hat

  def fit(self, x, y):
    self.train()
    self.optim.zero_grad()
    
    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    loss.backward()

    self.optim.step()

    loss_ = loss.detach().cpu().item()

    return loss_