# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    This function supports both legacy reward functions and the new Cross-View Spatial Reward (CVSR) system.
    
    CVSR implements: R = λ₁ * R_format + λ₂ * R_CVSR
    Where R_CVSR = w_ground * R_ground + w_overlap * R_overlap + w_ans * R_ans
    
    - R_format: Binary reward for output format correctness (think tags + boxed answer)
    - R_ground: IoU-based grounding reward using Hungarian algorithm  
    - R_overlap: Binary reward for correct cross-view overlap detection
    - R_ans: Task-specific answer correctness (binary for QA, distance-based for grasping)

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        sandbox_fusion_url (str, optional): URL for sandbox fusion (legacy parameter).
        concurrent_semaphore (optional): Semaphore for concurrent processing (legacy parameter).
        memory_limit_mb (int, optional): Memory limit in MB (legacy parameter).

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """

    # Updated legacy data sources to use CVSR
    if data_source in ['corl_l2_real_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.2, lambda2=0.8, w_ground=0.3, w_overlap=0.2, w_ans=0.5)
    elif data_source in ['corl_l2_real_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.3, lambda2=0.7, w_ground=0.1, w_overlap=0.1, w_ans=0.8)
    elif data_source in ['corl_l1_sim_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.3, lambda2=0.7, w_ground=0.1, w_overlap=0.2, w_ans=0.7)
    elif data_source in ['corl_l1_sim_train', 'corl_l1__sim_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.3, w_ans=0.3)
    elif data_source in ['corl_l3_sim_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.2, w_ans=0.4, d_max=50.0)
    elif data_source in ['corl_l3_sim_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.3, lambda2=0.7, w_ground=0.2, w_overlap=0.1, w_ans=0.7, d_max=50.0)
    elif data_source in ['corl_l2_sim_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.3, w_ans=0.3)
    elif data_source in ['corl_l2_sim_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.3, lambda2=0.7, w_ground=0.2, w_overlap=0.2, w_ans=0.6)
    elif data_source in ['corl_l3_real_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.2, lambda2=0.8, w_ground=0.3, w_overlap=0.2, w_ans=0.5, d_max=50.0)
    elif data_source in ['corl_l3_real_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.3, lambda2=0.7, w_ground=0.1, w_overlap=0.1, w_ans=0.8, d_max=50.0)
    
    # CVSR (Cross-View Spatial Reward) implementations
    elif data_source in ['corl_l1_merged_train', 'corl_l1_objectcount_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa", 
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.3, w_ans=0.3)
    elif data_source in ['corl_l1_merged_val', 'corl_l1_objectcount_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.3, lambda2=0.7, w_ground=0.2, w_overlap=0.2, w_ans=0.6)
    elif data_source in ['corl_l2_merged_train', 'corl_l2_spatial_understand_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.3, w_ans=0.3)
    elif data_source in ['corl_l2_merged_val', 'corl_l2_spatial_understand_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="qa",
                                lambda1=0.3, lambda2=0.7, w_ground=0.2, w_overlap=0.2, w_ans=0.6)
    elif data_source in ['corl_l3_merged_train', 'corl_l3_spatial_understand_train', 'corl_l3_grasping_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.2, lambda2=0.8, w_ground=0.4, w_overlap=0.2, w_ans=0.4, d_max=100.0)
    elif data_source in ['corl_l3_merged_val', 'corl_l3_spatial_understand_val', 'corl_l3_grasping_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.3, lambda2=0.7, w_ground=0.2, w_overlap=0.1, w_ans=0.7, d_max=100.0)
    elif data_source in ['corl_real_l3_grasping_train']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.2, lambda2=0.8, w_ground=0.3, w_overlap=0.2, w_ans=0.5, d_max=80.0)
    elif data_source in ['corl_real_l3_grasping_val']:
        from .cvsr_reward import compute_cvsr_reward
        res = compute_cvsr_reward(solution_str, ground_truth, task_type="grasping",
                                lambda1=0.3, lambda2=0.7, w_ground=0.1, w_overlap=0.1, w_ans=0.8, d_max=80.0)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
