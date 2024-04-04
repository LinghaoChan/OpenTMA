from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from .compute import ComputeMetrics, l2_norm, variance


class ComputeMetricsBest(ComputeMetrics):
    """
    This class is used to compute the best metrics. It extends the ComputeMetrics class.
    """
    def update(self, jts_text_: List[Tensor], jts_ref_: List[Tensor], lengths: List[List[int]]):
        """
        This method updates the metrics.

        Inputs:
        - jts_text_: a list of tensors representing the text
        - jts_ref_: a list of tensors representing the reference
        - lengths: a list of lists of integers representing the lengths

        Outputs: None
        """
        # Update the count and count_seq variables
        self.count += sum(lengths[0])
        self.count_seq += len(lengths[0])

        # Initialize the number of trials and the metrics list
        ntrials = len(jts_text_)
        metrics = []
        
        # Loop over each trial
        for index in range(ntrials):
            # Transform the text and reference tensors
            jts_text, poses_text, root_text, traj_text = self.transform(jts_text_[index], lengths[index])
            jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref_[index], lengths[index])

            # Initialize the metrics list for this trial
            mets = []
            
            # Loop over each length
            for i in range(len(lengths[index])):
                # Compute the root, pose, trajectory, and joints metrics
                APE_root = l2_norm(root_text[i], root_ref[i], dim=1).sum()
                APE_pose = l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
                APE_traj = l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
                APE_joints = l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

                # Compute the variance for the root, trajectory, poses, and joints
                root_sigma_text = variance(root_text[i], lengths[index][i], dim=0)
                root_sigma_ref = variance(root_ref[i], lengths[index][i], dim=0)
                AVE_root = l2_norm(root_sigma_text, root_sigma_ref, dim=0)

                traj_sigma_text = variance(traj_text[i], lengths[index][i], dim=0)
                traj_sigma_ref = variance(traj_ref[i], lengths[index][i], dim=0)
                AVE_traj = l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

                poses_sigma_text = variance(poses_text[i], lengths[index][i], dim=0)
                poses_sigma_ref = variance(poses_ref[i], lengths[index][i], dim=0)
                AVE_pose = l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

                jts_sigma_text = variance(jts_text[i], lengths[index][i], dim=0)
                jts_sigma_ref = variance(jts_ref[i], lengths[index][i], dim=0)
                AVE_joints = l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

                # Append the metrics to the metrics list for this trial
                met = [APE_root, APE_pose, APE_traj, APE_joints,
                       AVE_root, AVE_pose, AVE_traj, AVE_joints]
                mets.append(met)
            # Append the metrics for this trial to the overall metrics list
            metrics.append(mets)

        # Quick hacks
        mmm = metrics[np.argmin([x[0][0] for x in metrics])]
        APE_root, APE_pose, APE_traj, APE_joints, AVE_root, AVE_pose, AVE_traj, AVE_joints = mmm[0]
        self.APE_root += APE_root
        self.APE_pose += APE_pose
        self.APE_traj += APE_traj
        self.APE_joints += APE_joints
        self.AVE_root += AVE_root
        self.AVE_pose += AVE_pose
        self.AVE_traj += AVE_traj
        self.AVE_joints += AVE_joints
