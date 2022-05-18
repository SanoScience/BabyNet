import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time

from models.babynet import BabyNet
from video_data_loader import FetalWeightVideo
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="BabyNet for Fetal Birth Weight prediction")
parser.add_argument("--data",
                    type=str,
                    default="../data/",
                    help="Path to the data directory.")
parser.add_argument("--x_img_size",
                    type=int,
                    default=64,
                    help="Input X image size.")
parser.add_argument("--y_img_size",
                    type=int,
                    default=64,
                    help="Input Y image size")
parser.add_argument("--batch_size",
                    type=int,
                    default=2,
                    help="Number of batch size.")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="Number of epochs.")
parser.add_argument("--lr",
                    type=float,
                    default=0.0001,
                    help="Number of learning rate.")
parser.add_argument("--step_lr",
                    type=int,
                    default=16,
                    help="Step of learning rate")
parser.add_argument("--w_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay.")
parser.add_argument("--GPU",
                    type=bool,
                    default=True,
                    help="Use GPU.")
parser.add_argument("--display_steps",
                    type=int,
                    default=20,
                    help="Number of display steps.")
parser.add_argument("--model_name",
                    type=str,
                    default="BabyNet",
                    help="Name of trained model.")
parser.add_argument("--frames_num",
                    type=int,
                    default=16,
                    help="Number of frames in chunk")
parser.add_argument("--skip_frames",
                    type=int,
                    default=0,
                    help="Number of frames to skip")
parser.add_argument("--pixels_crop",
                    type=int,
                    default=0,
                    help="Number of frames in chunk")
parser.add_argument("--msha3D",
                    type=bool,
                    default=True,
                    help='Add MSHA to ResNet3D')
args = parser.parse_args()

dataset = FetalWeightVideo(input_path=args.data,
                           x_image_size=args.x_img_size,
                           y_image_size=args.y_img_size,
                           pixels_crop=args.pixels_crop,
                           skip_frames=args.skip_frames,
                           n_frames=args.frames_num)

if args.GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class CustomSequentialSampler(Sampler[int]):

    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self):
        for i in range(len(self.data_source)):
            yield self.data_source[i]

    def __len__(self) -> int:
        return len(self.data_source)


overlapping_ids_in_test = []


def ensure_no_patient_split(train_ids, valid_ids):
    patient_ids = dataset.patient_id_by_chunk
    patient_ids_train = set([patient_ids[i] for i in train_ids])
    patient_ids_valid = set([patient_ids[i] for i in valid_ids])
    overlapping_ids = patient_ids_train.intersection(patient_ids_valid)
    if len(overlapping_ids) == 0:
        return train_ids, valid_ids

    for overlapping_id in overlapping_ids:
        if overlapping_id not in overlapping_ids_in_test:
            indices_to_move = [ind for ind in train_ids if patient_ids[ind] == overlapping_id]
            valid_ids = np.append(valid_ids, indices_to_move).flatten()
            train_ids = np.delete(train_ids, np.searchsorted(train_ids, indices_to_move))
            overlapping_ids_in_test.append(overlapping_id)
        else:
            indices_to_move = [ind for ind in valid_ids if patient_ids[ind] == overlapping_id]
            train_ids = np.append(train_ids, indices_to_move).flatten()
            valid_ids = np.delete(valid_ids, np.searchsorted(valid_ids, indices_to_move))
    return list(sorted(train_ids)), list(sorted(valid_ids))


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.4f}")


kfold = KFold(n_splits=5, shuffle=False)
criterion_reg = nn.MSELoss()
loss_min = np.inf
train_dataset = FetalWeightVideo(input_path=args.data,
                                 x_image_size=args.x_img_size,
                                 y_image_size=args.y_img_size,
                                 pixels_crop=args.pixels_crop,
                                 skip_frames=args.skip_frames,
                                 n_frames=args.frames_num,
                                 mode="train")
val_dataset = FetalWeightVideo(input_path=args.data,
                               x_image_size=args.x_img_size,
                               y_image_size=args.y_img_size,
                               pixels_crop=args.pixels_crop,
                               skip_frames=args.skip_frames,
                               n_frames=args.frames_num,
                               mode="val")

print("---------------")

# Start time of learning
total_start_training = time.time()

for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    print(f"FOLD {fold}")
    print("----------------")
    train_ids, valid_ids = ensure_no_patient_split(train_ids, valid_ids)
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = CustomSequentialSampler(valid_ids)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_subsampler)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              sampler=valid_subsampler)

    model = BabyNet(msha=args.msha3D, n_frames=args.frames_num,
                    input_size=(args.y_img_size - args.pixels_crop, args.x_img_size - args.pixels_crop))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    scheduler = StepLR(optimizer=optimizer, step_size=args.step_lr, gamma=0.1, verbose=True)

    best_val_score = np.inf
    best_val_preds = None
    for epoch in range(args.epochs):
        start_time_epoch = time.time()
        print(f"Starting epoch {epoch + 1}")
        model.train()
        running_loss = 0.0

        y_true = []
        y_pred = []
        patient_running_loss = []
        for batch_idx, (videos, weights, patient_id, body_part, first_frame) in enumerate(train_loader):
            optimizer.zero_grad()
            videos = torch.permute(videos, (0, 4, 1, 2, 3))
            videos = videos.to(device=device).float()
            y_true.extend(weights.flatten().tolist())
            weights = weights.to(device=device).float()

            reg_out = model(videos)
            y_pred.extend(reg_out.flatten().cpu().tolist())
            loss_reg = criterion_reg(reg_out, weights)
            loss = loss_reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % args.display_steps == 0:
                print('    ', end='')
                print(f"Batch: {batch_idx + 1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} "
                      f"Learning time: {(time.time() - start_time_epoch):.2f}s "
                      f"First frame: {first_frame[0]}")

        # evalute
        calculate_metrics(y_true, y_pred)
        print(f"Finished epoch {epoch + 1}, starting evaluation.")

        model.eval()
        val_running_loss = 0.0
        y_true = []
        y_pred = []
        for batch_idx, (videos, weights, patient_id, body_part, first_frame) in enumerate(valid_loader):
            videos = torch.permute(videos, (0, 4, 1, 2, 3))
            videos = videos.to(device=device).float()
            y_true.extend(weights.flatten().tolist())
            weights = weights.to(device=device).float()

            reg_out = model(videos)
            y_pred.extend(reg_out.flatten().cpu().tolist())
            loss_reg = criterion_reg(reg_out, weights)
            loss = loss_reg

            val_running_loss += loss.item()

        calculate_metrics(y_true, y_pred)

        train_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(valid_loader)

        if best_val_score > val_loss:
            save_path = f"{args.model_name}-fold-{fold}.pt"
            torch.save(model.state_dict(), save_path)
            best_val_score = val_loss
            print(f"Current best val score {best_val_score}. Model saved!")

        scheduler.step()

        print('    ', end='')
        print(f"Train Loss: {train_loss:.3f} "
              f"Val Loss: {val_loss:.3f}")

print('Training finished, took {:.2f}s'.format(time.time() - total_start_training))
