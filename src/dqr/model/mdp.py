from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def compute_reward(t: int, relevance: int) -> int | float:
    """Compute reward function for MDP.

    In the original paper, "relevance" is called "rank."
    """
    if t == 0:
        return 0
    return relevance / np.log2(t + 1)


class State:
    def __init__(self, t: int, query: int, remaining: list[str]):
        self.t = t
        self.qid = query  # useful for sorting buffer
        self.remaining = remaining  # list of remaining documents to sort

    def pop(self) -> Any:
        return self.remaining.pop()

    def initial(self) -> bool:
        return self.t == 0

    def terminal(self) -> bool:
        return len(self.remaining) == 0


Batch = tuple[list[State], list[str], list[np.ndarray], list[State], list[bool]]


class BasicBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def push(
        self, state: State, action: Any, reward: float, next_state: State, done: bool
    ) -> None:
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def push_batch(self, df: pd.DataFrame, n: int) -> None:
        for i in range(n):
            random_qid = random.choice(list(df["qid"]))
            filtered_df = df.loc[df["qid"] == int(random_qid)].reset_index()
            row_order = [x for x in range(len(filtered_df))]
            X = [x[1]["doc_id"] for x in filtered_df.iterrows()]
            random.shuffle(row_order)
            for t, r in enumerate(row_order):
                cur_row = filtered_df.iloc[r]
                old_state = State(t, cur_row["qid"], X[:])
                action = cur_row["doc_id"]
                new_state = State(t + 1, cur_row["qid"], X[:])
                reward = compute_reward(t + 1, cur_row["rank"])
                self.push(old_state, action, reward, new_state, t + 1 == len(row_order))
                filtered_df.drop(filtered_df.index[[r]])

    def sample(self, batch_size: int) -> Batch:
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self) -> int:
        return len(self.buffer)
