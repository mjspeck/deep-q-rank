# External Packages
import argparse
import datetime as dt
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dqr.model.dqn import DQNAgent
from dqr.model.mdp import BasicBuffer
from dqr.util.preprocess import load_letor

warnings.simplefilter(action="ignore", category=UserWarning)

# BEGIN USER VARIABLES (CHANGE THESE)

TRAINING_SET_PATH = Path(__file__).parent / "../../../../MQ2008-list/Fold3/train.txt"
IS_TRAIN_SET_DIR = False  # Set to true if training on multiple sets in a directory
VAL_SET_PATH = Path(__file__).parent / "../../../../MQ2008-list/Fold3/vali.txt"
EPOCHS = 1000
OUTPUT_DIR = Path(os.getcwd()) / "output"
# OUTPUT_FILE_NAME = "/home/u27948/output/{}losses.txt".format(time.asctime())

# END USER VARIABLES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-path",
        "-tp",
        type=Path,
        default=TRAINING_SET_PATH,
        help="Path to training data file. Can be relative or absolute.",
    )
    parser.add_argument(
        "--validation-path",
        "-vp",
        type=Path,
        default=VAL_SET_PATH,
        help="Path to validation data file. Can be relative or absolute.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to save losses in.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print more logs.")
    parser.add_argument("--seed", "-s", type=int, help="Random seed.", default=2)
    parser.add_argument(
        "--epochs", "-e", type=int, help="Number of epochs", default=EPOCHS
    )
    parser.add_argument(
        "--batch-size", type=int, help="Number of samples per batch", default=1
    )

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Load in Data
    train_set = load_letor(args.training_path)
    val_set = load_letor(args.validation_path)
    session_path: Path = args.output_path / f"{dt.datetime.now()}__seed_{seed}".replace(
        " ", "_"
    )
    session_path.mkdir(parents=True, exist_ok=True)

    train_buffer = BasicBuffer(30000)
    train_buffer.push_batch(train_set, 3)

    val_buffer = BasicBuffer(20000)
    val_buffer.push_batch(val_set, 3)

    # Instantiate agent
    agent = DQNAgent(47, learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    # Begin Training
    y, z = [], []
    for i in tqdm(range(args.epochs)):
        # print("Beginning Iteration {}\n".format(i))
        y.append(agent.update(args.batch_size, verbose=args.verbose))
        z.append(
            agent.compute_loss(
                val_buffer.sample(args.batch_size), val_set, verbose=args.verbose
            )
        )

    # Save Model
    model_name = session_path / " model.pth"
    torch.save(agent.model.state_dict(), model_name)

    # Plot Losses
    y = [float(x) for x in y]
    z = [float(x) for x in z]
    plt.plot(z, label="val")
    plt.plot(y, label="train")
    plt.legend()
    plt.savefig(session_path / "losses.png")

    # Write Losses to File
    losses_path = session_path / "losses.txt"
    with open(losses_path, "w+") as f:
        f.write("Training Loss:\n")
        f.write(str(y))
        f.write("\n\n")
        f.write("Validation Loss:\n")
        f.write(str(z))
        f.write("\n")


if __name__ == "__main__":
    main()
