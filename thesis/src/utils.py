import os

import cv2
import nrrd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import FattyDataset
from matplotlib import pyplot as plt
from miseval import evaluate
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader


def data_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    test_dir,
    test_mask_dir,
    batch_size=4,
    train_transform=None,
    val_transform=None,
    test_transform=None,
):
    TR_FD = FattyDataset(
        image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform
    )
    VL_FD = FattyDataset(
        image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform
    )
    TS_FD = FattyDataset(
        image_dir=test_dir, mask_dir=test_mask_dir, transform=test_transform
    )

    train_loader = DataLoader(
        TR_FD, batch_size=batch_size, num_workers=4, shuffle=False
    )
    val_loader = DataLoader(
        VL_FD, batch_size=batch_size, num_workers=4, shuffle=False
    )
    test_loader = DataLoader(
        TS_FD, batch_size=batch_size, num_workers=4, shuffle=False
    )

    return train_loader, val_loader, test_loader


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(
                input.size(0), input.size(1), -1
            )  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(
                -1, input.size(2)
            )  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def save_preds(loader, model, model_path, device="cuda"):
    model = model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    # model.load_state_dict(torch.load(model_path))  # for cross validation pipeline
    model.eval()
    idx = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            # x = torch.squeeze(x, dim=1).to(device)
            y = torch.squeeze(y, dim=1).to(device)
            # y = y.to(device)
            y_pred = model(x)
            # y_pred = model(x[None, ...])
            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.squeeze(y, dim=0)
            y = torch.unsqueeze(y, dim=0)
            x = torch.squeeze(x, dim=0)
            y_pred = torch.squeeze(y_pred, dim=0)
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            nrrd.write(
                os.path.join(
                    os.getcwd(),
                    "..",
                    "data",
                    "seg_res",
                    "Proposed_model",
                    f"input_{idx}.nrrd",
                ),
                x,
            )
            nrrd.write(
                os.path.join(
                    os.getcwd(),
                    "..",
                    "data",
                    "seg_res",
                    "Proposed_model",
                    f"gt_{idx}.nrrd",
                ),
                y,
            )
            nrrd.write(
                os.path.join(
                    os.getcwd(),
                    "..",
                    "data",
                    "seg_res",
                    "Proposed_model",
                    f"seg_{idx}.nrrd",
                ),
                y_pred,
            )
            idx += 1


def visualize(dir):
    idx = 0
    for idx, file in enumerate(dir):
        # image, _ = nrrd.read(os.path.join(dir, f'input_{idx}.nrrd'), index_order='C')
        ground_truth, _ = nrrd.read(
            os.path.join(dir, f"gt_{idx}.nrrd"), index_order="C"
        )
        seg_image, _ = nrrd.read(
            os.path.join(dir, f"seg_{idx}.nrrd"), index_order="C"
        )
        # image = np.squeeze(image)
        ground_truth = np.squeeze(ground_truth)
        seg_image = np.squeeze(seg_image)
        final_image = ground_truth - seg_image
        print(ground_truth.shape)
        print(seg_image.shape)
        f, ax = plt.subplots(1, 3, figsize=(8, 8))
        ax[0].imshow(ground_truth)
        ax[0].set_title("Ground Truth", fontsize=16)
        ax[1].imshow(seg_image)
        ax[1].set_title("Segmented", fontsize=16)
        ax[2].imshow(final_image)
        ax[2].set_title("Final", fontsize=16)
        plt.show()
        idx += 1
        break
        # if idx == 5:
        #     break


def evaluation_metrics(loader, model, model_path, device="cuda"):
    model = model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    # model.load_state_dict(torch.load(model_path)) # For Cross validation
    model.eval()
    dice = dict()
    iou = dict()
    acc = dict()
    hd = dict()
    sens = dict()
    spec = dict()
    prec = dict()

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            # x = torch.squeeze(x, dim=1).to(device)
            y = torch.squeeze(y, dim=1).to(device)
            # y = y.to(device)
            y_pred = model(x)
            # y_pred = model(x[None, ...])
            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.squeeze(y, dim=0)
            x = torch.squeeze(x, dim=0)
            y_pred = torch.squeeze(y_pred, dim=0)
            pred = y_pred.cpu().numpy()
            label = y.cpu().numpy()
            dice[idx] = list(
                evaluate(
                    label, pred, metric="DSC", multi_class=True, n_classes=4
                )
            )
            iou[idx] = list(
                evaluate(
                    label, pred, metric="IoU", multi_class=True, n_classes=4
                )
            )
            acc[idx] = list(
                evaluate(
                    label, pred, metric="ACC", multi_class=True, n_classes=4
                )
            )
            hd[idx] = list(
                evaluate(
                    label, pred, metric="HD", multi_class=True, n_classes=4
                )
            )
            sens[idx] = list(
                evaluate(
                    label, pred, metric="SENS", multi_class=True, n_classes=4
                )
            )
            spec[idx] = list(
                evaluate(
                    label, pred, metric="SPEC", multi_class=True, n_classes=4
                )
            )
            prec[idx] = list(
                evaluate(
                    label, pred, metric="PREC", multi_class=True, n_classes=4
                )
            )

    return {
        "dice": dice,
        "iou": iou,
        "acc": acc,
        "hd": hd,
        "sens": sens,
        "spec": spec,
        "prec": prec,
    }
