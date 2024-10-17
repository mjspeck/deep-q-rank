# External Packages
import argparse
import time

import torch

from dqr.model.dqn import DQNAgent
from dqr.model.mdp import BasicBuffer
from dqr.util.eval import get_just_tau
from dqr.util.preprocess import load_letor

# BEGIN USER VARIABLES (CHANGE THESE)

LETOR_PATH = "/home/u27948/data"
OUTPUT_FILE_NAME = "./logs/" + time.asctime() + " tau error output.txt"
EPOCHS = 10000

# END USER VARIABLES


def train_model(fold):
    # Load in Data
    train_set = load_letor(LETOR_PATH + "/Fold{}/train.txt".format(fold))
    train_buffer = BasicBuffer(100000)
    while len(train_buffer) < 100000:
        train_buffer.push_batch(train_set, 1)

    # Instantiate agent
    agent = DQNAgent((47,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    # Begin Training
    for i in range(EPOCHS):
        if i % 100 == 0:
            print("Beginning Iteration {}\n".format(i))
        agent.update(1, verbose=False)

    # Save Model
    model_name = time.asctime() + " fold{} trained model.pth".format(fold)
    torch.save(agent.model.state_dict(), model_name)
    print("Saved Model for fold{}".format(fold))

    # Get Training Errors
    error_list = get_just_tau(agent, train_set)
    print("Fold {} train error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Train Tau Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))

    return agent


def val_model(agent, fold):
    # Load in Data
    val_set = load_letor(LETOR_PATH + "/Fold{}/vali.txt".format(fold))
    val_buffer = BasicBuffer(100000)
    while len(val_buffer) < 100000:
        val_buffer.push_batch(val_set, 1)

    error_list = get_just_tau(agent, val_set)
    print("Fold {} val error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Val Tau Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))


def test_model(agent, fold):
    # Load in Data
    test_set = load_letor(LETOR_PATH + "/Fold{}/test.txt".format(fold))
    test_buffer = BasicBuffer(100000)
    while len(test_buffer) < 100000:
        test_buffer.push_batch(test_set, 1)

    error_list = get_just_tau(agent, test_set)
    print("Fold {} val error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Test Tau Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))

    print("Finished Successfully Evaluating Model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fold value.")
    parser.add_argument("--fold", help="fold help")
    args = parser.parse_args()
    FOLD = args.fold
    agent = train_model(FOLD)
    val_model(agent, FOLD)
    test_model(agent, FOLD)
