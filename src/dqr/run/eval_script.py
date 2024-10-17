# External Packages

import time

import torch

from dqr.model.dqn import DQN, DQNAgent
from dqr.util.eval import eval_agent_final
from dqr.util.preprocess import load_letor

# BEGIN USER VARIABLES (CHANGE THESE)

LETOR_PATH = "/home/u27948/data"
OUTPUT_FILE_NAME = time.asctime() + " eval output.txt"
PRETRAINED_MODEL_PATH = "/home/u27948/deep-q-rank/best_model.pth"
NDCG_K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# END USER VARIABLES


def train_model(LETOR_PATH, OUTPUT_FILE_NAME, PRETRAINED_MODEL_PATH):
    start_time = time.time()
    print("Creating Agent from Saved Model")
    model = DQN((47,), 1)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    agent = DQNAgent(
        (47,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model
    )

    for fold in [2, 3, 4, 5]:
        print("\nRunning Fold {}".format(fold))
        print("Loading LETOR TEST SET")
        test_path = LETOR_PATH + "/Fold{}".format(fold) + "/test.txt"
        letor_test = load_letor(test_path)
        print("Running Eval on LETOR Test")
        ndcg_list = eval_agent_final(agent, NDCG_K_LIST, letor_test)
        print("Saving results")
        with open(OUTPUT_FILE_NAME, "a+") as f:
            f.write("Fold {} NDCG Values: {}\n".format(fold, NDCG_K_LIST))
            f.write(str(ndcg_list))
            f.write("\n")

    print("Finished Successfully Evaluating Model.")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    train_model(LETOR_PATH, OUTPUT_FILE_NAME, PRETRAINED_MODEL_PATH)
