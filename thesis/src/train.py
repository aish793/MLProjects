import gc
import json
import os

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from model import Att_ResUnet
from utils import (
    FocalLoss,
    data_loaders,
    evaluation_metrics,
    save_preds,
    visualize,
)

TRAIN_IMAGE_DIR = os.path.join(os.getcwd(), "..", "data", "train_images")
TRAIN_MASK_DIR = os.path.join(os.getcwd(), "..", "data", "train_masks")
VALID_IMAGE_DIR = os.path.join(os.getcwd(), "..", "data", "valid_images")
VALID_MASK_DIR = os.path.join(os.getcwd(), "..", "data", "valid_masks")
TEST_IMAGE_DIR = os.path.join(os.getcwd(), "..", "data", "test_images")
TEST_MASK_DIR = os.path.join(os.getcwd(), "..", "data", "test_masks")
#
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.3),
        # A.Transpose(p=0.3),
        # A.RandomRotate90(p=0.4),
        A.GridDistortion(p=0.4),
        # ToTensorV2(),
    ],
)
# #
train_loader, val_loader, test_loader = data_loaders(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    batch_size=1,
    train_transform=train_transform,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = Att_ResUnet()
model_save_pth = os.path.join(os.getcwd(), "Att_resunet_FL.pth")


def train(train_loader, val_loader, model, model_name):
    print(f"Training Model {model_name} ...")
    model = model.to(device=DEVICE)
    # loss_criterion = nn.CrossEntropyLoss()
    loss_criterion = FocalLoss()

    optimizer = torch.optim.Adam(params=model.parameters())

    epochs = 100
    model.train()

    for epoch in range(epochs):
        epoch_train_loss = 0
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for train_images, train_masks in train_loader:
                # forward
                train_images = train_images.to(device=DEVICE)
                train_masks = torch.squeeze(train_masks, dim=1).to(
                    device=DEVICE
                )
                # Cross Entropy loss takes the target containing class indices, so channel has to be removed
                preds = model(train_images)
                train_masks = torch.squeeze(train_masks, dim=1)  # to change
                loss = loss_criterion(preds, train_masks)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.detach().item()
                pbar.update(1)

        print("Epoch: ", epoch, f"Training loss: {epoch_train_loss}")

        epoch_val_loss = 0
        model.eval()
        with tqdm.tqdm(total=len(val_loader)) as pbar:
            for valid_images, valid_masks in val_loader:
                valid_images = valid_images.to(device=DEVICE)
                valid_masks = torch.squeeze(valid_masks, dim=1).to(
                    device=DEVICE
                )
                valid_preds = model(valid_images)
                valid_masks = torch.squeeze(valid_masks, dim=1)
                val_loss = loss_criterion(valid_preds, valid_masks)
                epoch_val_loss += val_loss.detach().item()
                pbar.update(1)

            print("Epoch: ", epoch, ", Validation loss: ", epoch_val_loss)

    gc.collect()
    torch.save({"state_dict": model.state_dict()}, model_save_pth)


train(train_loader, val_loader, model, "r2unet_new")


##########################Evaluation###########################################
metrics = evaluation_metrics(
    loader=test_loader, model=model, model_path=model_save_pth
)
import json

with open(os.path.join(os.getcwd(), "metrics.json"), "w") as jsonfile:
    json.dump(metrics, jsonfile, indent=4)
    print("JSON file saved successfully.")

with open(os.path.join(os.getcwd(), "metrics.json"), "r") as jsonfile:
    metrics = json.load(jsonfile)

for name, results in metrics.items():
    df = pd.DataFrame.from_dict(results).T
    df["mean_dice"] = df.mean(axis=1)
    filename = f"model_{name}.xlsx"
    df.to_excel(filename)
    print("Done")

save_preds(loader=test_loader, model=model, model_path=model_save_pth)
visualize(dir=os.path.join(os.getcwd(), "data", "Proposed_model"))
