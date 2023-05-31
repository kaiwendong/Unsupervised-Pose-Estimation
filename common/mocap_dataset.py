# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from common.skeleton import Skeleton
h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

import numpy as np

class MocapDataset:
    def __init__(self, fps=50, skeleton=h36m_skeleton):
        self._data = None # Must be filled by subclass
        self._cameras = None # Must be filled by subclass
        self._skeleton = skeleton
        self._fps = fps
    
    def __getitem__(self, key):
        return self._data[key]
        
    def subjects(self):
        return self._data.keys()
        
    def cameras(self):
        return self._cameras

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                s['positions'] = s['positions'][:, kept_joints]
