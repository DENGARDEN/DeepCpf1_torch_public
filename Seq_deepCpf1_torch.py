# Reference: https://github.com/MyungjaeSong/Paired-Library
# PyTorch implementation
import argparse
import os

# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

from deepcpf1_network import SequenceDataset
import deepcpf1_network


def main():
    # TODO: Update Usage
    print(
        "Usage: python Seq_deepCpf1_torch.py --train ./data/train.csv --test ./data/test.csv -- output output.csv"
    )
    print("input.txt must include 3 columns with single header row")
    print("\t1st column: sequence index")
    print("\t2nd column: 34bp target sequence")
    print("\t3rd column: binary chromain information of the target sequence\n")

    print("DeepCpf1 currently requires python=3.9, PyTorch=1.12")
    print(
        "DeepCpf1 available on GitHub requires pre-obtained binary chromatin information (DNase-seq narraow peak data from ENCODE)"
    )
    print(
        "DeepCpf1 web tool, available at http://data.snu.ac.kr/DeepCpf1, provides entire pipeline including binary chromatin accessibility for 125 cell lines\n"
    )

    # if len(sys.argv) < 3:
    #     print("ERROR: Not enough arguments for DeepCpf1.py; Check the usage.")
    #     sys.exit()

    # Argument Parsing
    parser = argparse.ArgumentParser("DeepCpf1 meets PyTorch")
    parser.add_argument("--train", type=str, default="./data/train.csv")
    parser.add_argument("--test", type=str, default="./data/test.csv")
    parser.add_argument("--load_weights", action="store_true", default=True)
    parser.add_argument("--model_path", type=str, default="weights")
    parser.add_argument(
        "--mps", action="store_true", default=False, help="Apple Silicon MPS"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--training_mode", action="store_true", help="training")
    parser.add_argument(
        "--sequence_length", type=int, default=34, help="target sequence length"
    )
    parser.add_argument("--kernel_size", type=int, default=5, help="kernel size")
    parser.add_argument("--pool_size", type=int, default=2, help="pooling filter size")
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="maximum training epoch"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="Saving network state_dict",
    )
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Model save frequency"
    )
    parser.add_argument("--alias", type=str, default="corr_testing")

    args = parser.parse_args()

    # Device Configuration
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": int(1e4), "shuffle": True}
    test_kwargs = {"batch_size": 50, "shuffle": True}

    seq_deep_cpf1 = deepcpf1_network.SeqDeepCpf1Net(args).to(device)
    opt_seq_deep_cpf1 = optim.Adam(seq_deep_cpf1.parameters(), lr=0.001)

    train_history = []
    model_state_paths = []

    if args.training_mode:
        # -----------------------------------------------------
        # Seq-deepCpf1 PreTrain
        # -----------------------------------------------------
        print("Training Seq-deepCpf1")
        print("Loading train data")

        # Load train data
        training_data = SequenceDataset(csv_file=args.train, args=args)
        train_dataloader = DataLoader(training_data, **train_kwargs)

        # Assertive
        epoch = 1
        while epoch <= args.max_epoch:
            loss = deepcpf1_network.train(
                args, seq_deep_cpf1, device, train_dataloader, opt_seq_deep_cpf1, epoch
            )
            train_history.append(loss)
            epoch += 1

        epochs = [i for i in range(1, args.max_epoch + 1)]
        plt.plot(epochs, train_history, "g", label="Training loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if args.load_weights:
        # Loading Model for Inference
        print("Listing weights for the models")
        for file in os.listdir(args.model_path):
            if file.endswith(".pt"):
                model_state_paths.append(os.path.join(args.model_path, file))

        seq_deep_cpf1.eval()
    print("Loading test data")

    # TODO: TESTING
    test_history = []
    # Load test data
    testing_data = SequenceDataset(csv_file=args.test, args=args)
    test_dataloader = DataLoader(testing_data, **test_kwargs)

    for idx, model_path in enumerate(model_state_paths):
        seq_deep_cpf1.load_state_dict(torch.load(model_path))

        print(f"Predicting on test data: {idx}/{len(model_state_paths)}")
        Seq_deepCpf1_SCORE = deepcpf1_network.test(
            seq_deep_cpf1, device, test_dataloader
        )  # returns average loss
        test_history.append(Seq_deepCpf1_SCORE)

    # Plotting test error over the generated models
    loss_test = test_history
    epochs = [i for i in range(1, args.max_epoch + 1, args.save_freq)]
    plt.plot(epochs, loss_test, "b", label="Testing loss")

    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
