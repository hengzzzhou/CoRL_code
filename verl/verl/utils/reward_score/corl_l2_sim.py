"""
CORL Level 2 Simulation Reward - Updated to use CVSR Framework
"""

from .cvsr_reward import compute_cvsr_reward


def compute_reward(solution_str: str, ground_truth: str, 
                  partial_credit: bool = False, bbox_reward_weight: float = 0.3) -> float:
    """
    Compute reward for CORL Level 2 simulation tasks using CVSR framework.
    
    This function provides backward compatibility while using the new CVSR system.
    The bbox_reward_weight parameter is mapped to CVSR component weights.
    
    Args:
        solution_str: Model's output string
        ground_truth: Ground truth string
        partial_credit: Compatibility parameter (not used in CVSR)
        bbox_reward_weight: Weight for bounding box component (mapped to w_ground)
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Map legacy bbox_reward_weight to CVSR weights for simulation tasks
    if bbox_reward_weight > 0:
        # Training mode: emphasize spatial reasoning components
        w_ground = min(0.5, bbox_reward_weight * 1.3)  # Scale grounding importance
        w_overlap = 0.3  # Important for cross-view spatial understanding
        w_ans = 1.0 - w_ground - w_overlap
        lambda1, lambda2 = 0.2, 0.8
    else:
        # Validation mode: emphasize final answer accuracy
        w_ground = 0.2
        w_overlap = 0.2
        w_ans = 0.6
        lambda1, lambda2 = 0.3, 0.7
    
    return compute_cvsr_reward(
        solution_str, ground_truth, 
        task_type="qa",
        lambda1=lambda1, lambda2=lambda2,
        w_ground=w_ground, w_overlap=w_overlap, w_ans=w_ans
    )


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Legacy interface for compute_score."""
    return compute_reward(solution_str, ground_truth, partial_credit=True, bbox_reward_weight=0.3)