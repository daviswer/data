# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from typing import Any, Dict, Iterator, Optional

import torch
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.loader import Loader


class MockGenerator:
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


def MockSource(num_samples: int) -> BaseNode[dict]:
    return IterableWrapper(MockGenerator(num_samples))


def udf_raises(item):
    raise ValueError("test exception")


class RandomSleepUdf:
    def __init__(self, sleep_max_sec: float = 0.01) -> None:
        self.sleep_max_sec = sleep_max_sec

    def __call__(self, x):
        time.sleep(random.random() * self.sleep_max_sec)
        return x


class Collate:
    def __call__(self, x):
        result = {}
        for k in x[0].keys():
            result[k] = [i[k] for i in x]
        return result


class IterInitError(BaseNode[int]):
    def __init__(self, msg: str = "Iter Init Error") -> None:
        super().__init__()
        self.msg = msg

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        raise ValueError(self.msg)

    def next(self):
        raise ValueError("next() should not be called")

    def get_state(self) -> Dict[str, Any]:
        return {}


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_samples: int, name: str) -> None:
        self.num_samples = num_samples
        self.name = name

    def __iter__(self) -> Iterator[dict]:
        for i in range(self.num_samples):
            yield {
                "name": self.name,
                "step": i,
                "test_tensor": torch.tensor([i]),
                "test_str": f"str_{i}",
            }


class DummyMapDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> dict:
        return {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


def run_test_save_load_state(test, node: BaseNode, midpoint: int):
    ##############################
    # Generate initial, midpoint, and end state_dict's
    x = Loader(node)

    initial_state_dict = x.state_dict()
    it = iter(x)
    results = []
    for _ in range(midpoint):
        results.append(next(it))
    state_dict = x.state_dict()
    for val in it:
        results.append(val)

    state_dict_0_end = x.state_dict()

    # store epoch 1's results
    it = iter(x)
    results_1 = []
    for _ in range(midpoint):
        results_1.append(next(it))
    state_dict_1 = x.state_dict()
    for val in it:
        results_1.append(val)

    ##############################
    # Test restoring from midpoint
    x.load_state_dict(state_dict)
    results_after = list(x)
    test.assertEqual(results_after, results[midpoint:])

    # Test for second epoch after resume
    results_after_1 = list(x)
    test.assertEqual(results_after_1, results_1)

    ##############################
    # Test restoring from midpoint of epoch 1
    x.load_state_dict(state_dict_1)
    results_after_2 = list(x)
    test.assertEqual(results_after_2, results_1[midpoint:])

    ##############################
    # Test initialize from beginning after resume
    x.load_state_dict(initial_state_dict)
    full_results = list(x)
    test.assertEqual(full_results, results)
    full_results_1 = list(x)
    test.assertEqual(full_results_1, results_1)

    ##############################
    # Test restoring from end-of-epoch 0
    x = Loader(node, restart_on_stop_iteration=False)
    x.load_state_dict(state_dict_0_end)
    results_after_dict_0_with_restart_false = list(x)
    test.assertEqual(results_after_dict_0_with_restart_false, [])

    ##############################
    # Test restoring from end of epoch 0 with restart_on_stop_iteration=True
    x = Loader(node)
    x.load_state_dict(state_dict_0_end)
    results_after_dict_0 = list(x)
    test.assertEqual(results_after_dict_0, results_1)
