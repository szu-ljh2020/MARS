import functools
import numpy as np
import torch
from torch_geometric.data import Data


class BeamSearchNode(object):
    def __init__(self, hiddenstate=None, logProb=0., input_next=[0, ], p_smi='', type=None):
        self.h = hiddenstate
        self.logp = logProb
        self.log_probs = []
        self.input_next = input_next
        self.edge_transformation = []
        self.atom_transformation = []
        self.transformation_paths = []
        self.targets_predict = []
        self.attachments_list = []
        self.active_attachments = 0
        self.terminatation = False
        self.p_smi = p_smi
        self.synthon = Data(x=torch.tensor(np.zeros(shape=(1, 45)), dtype=torch.float32),
                            edge_index=torch.empty((2, 0), dtype=torch.long),
                            edge_attr=torch.empty((0, 12), dtype=torch.bool),
                            type=type)

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class PriorityQueue(object):
    def __init__(self, max_size=100, values=[]):
        self.max_size = max_size
        self.compare = functools.cmp_to_key(lambda x, y: x[0] - y[0])
        self.values = sorted(values, key=self.compare, reverse=True)

    def add(self, value, sort=False):
        self.values.append(value)
        if sort:
            self.values = sorted(self.values, key=self.compare, reverse=True)

    def fit_size(self):
        values = sorted(self.values, key=self.compare, reverse=True)
        self.values = []
        finished_predictions = set()
        # remove duplicated finished predictions
        for value in values:
            if value[1].input_next[0][0] == 7:
                # arrange transformation in canonical order
                prev = 0
                transform_paths = []
                for idx, transform in enumerate(value[1].transformation_paths):
                    if transform[0] and idx > 0:
                        transform_paths.append(value[1].transformation_paths[prev:idx])
                        prev = idx
                transform_paths.append(value[1].transformation_paths[prev:])
                attachments = [tp[0][-1] for tp in transform_paths]
                indexes = np.argsort(attachments)[::-1]
                transformation_paths_sorted = []
                for idx in indexes:
                    transformation_paths_sorted.extend(transform_paths[idx])
                value[1].transformation_paths = transformation_paths_sorted
                transformation_paths_sorted = tuple([tuple(tp) for tp in transformation_paths_sorted])
                if transformation_paths_sorted not in finished_predictions:
                    finished_predictions.add(transformation_paths_sorted)
                    self.values.append(value)
            else:
                self.values.append(value)
        self.values = self.values[:self.max_size]

    def size(self):
        return len(self.values)