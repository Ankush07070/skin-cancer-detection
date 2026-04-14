import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.logger import logging
from src.exception import CustomException
import sys

def train_one_epoch(model, loader, optimizer, device):
    try:
        model.train()
        total_loss = 0

        criterion = nn.CrossEntropyLoss()

        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    except Exception as e:
        logging.error(f"Error in train_one_epoch: {str(e)}")
        raise CustomException(e, sys)


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total