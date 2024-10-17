# External Packages
import random
import time

import matplotlib.pyplot as plt
import torch

from dqr.model.dqn import DQNAgent
from dqr.model.mdp import BasicBuffer
from dqr.util.preprocess import load_letor

random.seed(2)

# BEGIN USER VARIABLES (CHANGE THESE)

TRAINING_SET_PATH = "/home/u27948/data/Fold3/train.txt"
IS_TRAIN_SET_DIR = False  # Set to true if training on multiple sets in a directory
VAL_SET_PATH = "/home/u27948/data/Fold3/vali.txt"
EPOCHS = 1000
OUTPUT_FILE_NAME = "/home/u27948/output/{}losses.txt".format(time.asctime())

# END USER VARIABLES


def main():
    # Load in Data
    train_set = load_letor(TRAINING_SET_PATH)
    val_set = load_letor(VAL_SET_PATH)

    train_buffer = BasicBuffer(30000)
    train_buffer.push_batch(train_set, 3)

    val_buffer = BasicBuffer(20000)
    val_buffer.push_batch(val_set, 3)

    # Instantiate agent
    agent = DQNAgent((47,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    # Begin Training
    y, z = [], []
    for i in range(EPOCHS):
        print("Beginning Iteration {}\n".format(i))
        y.append(agent.update(1, verbose=True))
        z.append(agent.compute_loss(val_buffer.sample(1), val_set, verbose=True))

    # Save Model
    model_name = time.asctime() + " model.pth"
    torch.save(agent.model.state_dict(), model_name)

    # Plot Losses
    y = [float(x) for x in y]
    z = [float(x) for x in z]
    plt.plot(z, label="val")
    plt.plot(y, label="train")
    plt.legend()
    plt.savefig("foo.png")

    # Write Losses to File
    with open(OUTPUT_FILE_NAME, "w+") as f:
        f.write("Training Loss:\n")
        f.write(str(y))
        f.write("\n\n")
        f.write("Validation Loss:\n")
        f.write(str(z))
        f.write("\n")


if __name__ == "__main__":
    main()
