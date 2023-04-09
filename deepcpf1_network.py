import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Parameters


def _preprocess(seq, seq_len):
    # custom one-hot encoding
    DATA_X = torch.zeros(1, seq_len, 4, dtype=torch.float)

    # data == [num, 34 bp sequence, indel_freq, CA]

    for i in range(seq_len):
        if seq[i] in "Aa":
            DATA_X[0, i, 0] = 1
        elif seq[i] in "Cc":
            DATA_X[0, i, 1] = 1
        elif seq[i] in "Gg":
            DATA_X[0, i, 2] = 1
        elif seq[i] in "Tt":
            DATA_X[0, i, 3] = 1
        else:
            print("Non-ATGC character " + seq[i])
            sys.exit()
    return DATA_X


def _postprocess(vec, seq_len):
    # custom one-hot encoding vector to original sequence
    # example dimension (1,29,4)
    target_seq = list()

    for i in range(seq_len):
        if vec[0, i, 0] == 1:
            target_seq.append("A")
        elif vec[0, i, 1] == 1:
            target_seq.append("C")
        elif vec[0, i, 2] == 1:
            target_seq.append("G")
        elif vec[0, i, 3] == 1:
            target_seq.append("T")

    original_seq = "".join([str(nt) for nt in target_seq])

    return original_seq


def decoding(vectors, seq_len: int):
    # Sample input dimension (522, 1, 29, 4)
    decoded = list()

    for vec in vectors:
        decoded.append(_postprocess(vec, seq_len))

    decoded = np.array(decoded)

    return decoded


# Reference: https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
class SequenceDataset(Dataset):
    """Sequence dataset"""

    def __init__(
        self,
        csv_file,
        args,
        transform=None,
        target_transform=None,
        is_test=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.indel_frame = pd.read_csv(csv_file)
        print(self.indel_frame)
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        self.args = args

        # data preprocessing
        # [original target sequence, one-hot encoded vector, indel frequency, CA]
        processed_dataset = {"Origin": [], "Vector": [], "Indel": [], "CA": []}

        # [target_sequence, indel frequency, CA]
        cols = list(self.indel_frame.columns)
        for index, row in self.indel_frame.iterrows():
            raw_sequence = row.get(cols[0])
            indel_freq = row.get(cols[1])
            ca = row.get(cols[2])

            processed_dataset["Origin"].append(raw_sequence)
            processed_dataset["Vector"].append(
                _preprocess(raw_sequence, self.args.sequence_length)
            )
            processed_dataset["Indel"].append(indel_freq)
            processed_dataset["CA"].append(ca)

        self.processed_df = pd.DataFrame(
            processed_dataset, columns=["Origin", "Vector", "Indel", "CA"]
        )

    def __len__(self):
        return len(self.indel_frame)

    def __getitem__(self, idx):
        # In 'train.csv', the columns should be target sequence, indel frequency, chromatin accessibility

        if torch.is_tensor(idx):
            idx = idx.tolist()
        original_sequence = self.processed_df.iloc[idx, 0]
        seq = self.processed_df.iloc[idx, 1]
        ca = self.processed_df.iloc[idx, 3]

        label = self.processed_df.iloc[idx, 2]
        label = np.array([label]).astype("float")
        label = torch.from_numpy(label)

        return (seq.float(), ca, original_sequence), label.float()


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class SeqDeepCpf1Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Adopting DeepCpf1 NN structure
        self.Seq_deepCpf1_C1 = nn.Conv2d(
            in_channels=1, out_channels=80, kernel_size=(args.kernel_size, 4)
        )
        torch.nn.init.xavier_uniform(self.Seq_deepCpf1_C1.weight)
        self.Seq_deepCpf1_C1.bias.data.fill_(0.0)
        w_ = (args.sequence_length - args.kernel_size + 0) // 1 + 1  # After CONV
        w_ = (w_ - args.pool_size) // args.pool_size + 1  # After POOL
        self.Seq_deepCpf1_P1 = torch.nn.AdaptiveAvgPool2d((w_, 1))
        self.Seq_deepCpf1_DO1 = nn.Dropout(0.3)
        # 4 * 14 * 80 (flattened)

        dim = w_ * 80
        self.Seq_deepCpf1_D1 = nn.Linear(in_features=dim, out_features=80)
        self.Seq_deepCpf1_DO2 = nn.Dropout(0.3)
        self.Seq_deepCpf1_D2 = nn.Linear(in_features=80, out_features=40)
        self.Seq_deepCpf1_DO3 = nn.Dropout(0.3)
        self.Seq_deepCpf1_D3 = nn.Linear(in_features=40, out_features=40)
        self.Seq_deepCpf1_DO4 = nn.Dropout(0.3)
        self.Seq_deepCpf1_Output = nn.Linear(in_features=40, out_features=1)

    def forward(self, x, _):
        # Input matrix.dim == 34 * 4 one-hot encoding matrix
        x = F.relu(self.Seq_deepCpf1_C1(x))
        x = self.Seq_deepCpf1_P1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.Seq_deepCpf1_DO1(x)
        x = F.relu(self.Seq_deepCpf1_D1(x))
        x = self.Seq_deepCpf1_DO2(x)
        x = F.relu(self.Seq_deepCpf1_D2(x))
        x = self.Seq_deepCpf1_DO3(x)
        x = F.relu(self.Seq_deepCpf1_D3(x))
        x = self.Seq_deepCpf1_DO4(x)

        return self.Seq_deepCpf1_Output(x)


# Reference : https://github.com/pytorch/examples/blob/78acb79062189bd937f17edf7c97571e6ec59083/mnist/main.py#L97
def train(args, model, device, train_loader, optimizer, epoch, fold=None):
    model.train()
    loss = 0

    fold = fold if fold is not None else -1
    for batch_idx, (data, target) in enumerate(train_loader):
        # data == (seq, ca)
        seq, ca, target = (
            data[0].to(device),
            data[1].to(device),
            target.to(device),
        )
        optimizer.zero_grad()
        output = model(seq, ca)
        loss = F.mse_loss(output, target).to(dtype=torch.float32)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f"CV fold: {fold}; -1 means no CV")
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(seq),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # Saving Loading Model for Inference
    if epoch % args.save_freq == 0 and args.save_model is True:
        torch.save(
            model.state_dict(),
            f"{args.model_path}/{args.alias}_{model.__class__.__name__}_e{epoch}_fold{fold}_state.pt",
        )
        return loss.item()


def test(model, device, test_loader, fold=None):
    model.eval()
    test_loss = 0

    fold = fold if fold is not None else -1
    with torch.no_grad():
        for data, target in test_loader:
            seq, ca, target = data[0].to(device), data[1].to(device), target.to(device)
            output = model(seq, ca)
            test_loss += F.mse_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

    # Regression model
    test_loss /= len(test_loader.dataset)

    print(f"CV fold: {fold}; -1 means no CV")
    print("\nTest set: Average loss: {:.4f},\n".format(test_loss))

    return test_loss


# TODO
def predict(model, device, test_loader):
    # https://discuss.pytorch.org/t/efficient-method-to-gather-all-predictions/8008/5
    model.eval()

    sequence_vectors = torch.tensor([], dtype=torch.float, device=device)
    y_true = torch.tensor([], dtype=torch.float, device=device)
    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for data, targets in test_loader:
            seq, ca, targets = (
                data[0].to(device),
                data[1].to(device),
                targets.to(device),
            )

            outputs = model(seq, ca)

            sequence_vectors = torch.cat((sequence_vectors, seq), 0)
            y_true = torch.cat((y_true, targets), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    sequence_vectors = sequence_vectors.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = all_outputs.cpu().numpy()

    # Regression model
    return sequence_vectors, y_true, y_pred
