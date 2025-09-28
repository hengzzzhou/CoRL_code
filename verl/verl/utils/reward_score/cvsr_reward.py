"""
Cross-View Spatial Reward (CVSR) Design Implementation

This module implements the Cross-View Spatial Reward system as described in the paper:
R = λ₁ * R_format + λ₂ * R_CVSR

Where R_CVSR = w_ground * R_ground + w_overlap * R_overlap + w_ans * R_ans
"""

import re
import math
from typing import Union, Dict, Any, Optional, List, Tuple
import json


class CVSRReward:
    """
    Cross-View Spatial Reward (CVSR) implementation for dual-robot perspective tasks.
    
    Implements the reward function:
    R = λ₁ * R_format + λ₂ * R_CVSR
    
    Where:
    - R_format: Binary reward for output format correctness (think tags + boxed answer)
    - R_CVSR: Cross-view spatial reward with three components:
      - R_ground: IoU-based grounding reward using Hungarian algorithm
      - R_overlap: Binary reward for correct cross-view overlap detection
      - R_ans: Task-specific answer correctness
    """
    
    def __init__(self, 
                 lambda1: float = 0.2,  # Weight for format reward
                 lambda2: float = 0.8,  # Weight for CVSR 
                 w_ground: float = 0.3,  # Weight for grounding reward
                 w_overlap: float = 0.2,  # Weight for overlap accuracy
                 w_ans: float = 0.5,     # Weight for answer correctness
                 d_max: float = 100.0,   # Max distance for grasping normalization
                 task_type: str = "qa"):  # Task type: "qa" or "grasping"
        """
        Initialize the CVSR reward model.
        
        Args:
            lambda1: Weight for output format reward (λ₁)
            lambda2: Weight for CVSR component (λ₂) 
            w_ground: Weight for grounding reward component
            w_overlap: Weight for overlap accuracy component
            w_ans: Weight for answer correctness component
            d_max: Maximum distance for grasping task normalization
            task_type: Type of task ("qa" for Q&A, "grasping" for coordinate prediction)
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w_ground = w_ground
        self.w_overlap = w_overlap
        self.w_ans = w_ans
        self.d_max = d_max
        self.task_type = task_type
        
        # Ensure weights sum to 1.0 for CVSR components
        total_w = w_ground + w_overlap + w_ans
        if total_w > 0:
            self.w_ground = w_ground / total_w
            self.w_overlap = w_overlap / total_w  
            self.w_ans = w_ans / total_w
        
        # Ensure lambda weights sum to 1.0
        total_lambda = lambda1 + lambda2
        if total_lambda > 0:
            self.lambda1 = lambda1 / total_lambda
            self.lambda2 = lambda2 / total_lambda

    def compute_r_format(self, output_str: str) -> float:
        """
        Compute R_format: Binary reward for structural correctness.
        
        The model receives R_format=1 if:
        1. Intermediate reasoning is enclosed in <think>...</think> tags
        2. Final answer appears in \\boxed{} format
        
        Args:
            output_str: The model's output string
            
        Returns:
            Binary reward (0.0 or 1.0)
        """
        # Check for <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        has_think = bool(re.search(think_pattern, output_str, re.DOTALL))
        
        # Check for \\boxed{} format
        boxed_patterns = [
            r'\\\\boxed\{([^}]*)\}',  # \\boxed{answer}
            r'\\boxed\{([^}]*)\}',     # \boxed{answer}
            r'\$\\boxed\{([^}]*)\}\$', # $\boxed{answer}$
        ]
        
        has_boxed = False
        for pattern in boxed_patterns:
            if re.search(pattern, output_str):
                has_boxed = True
                break
                
        return 1.0 if (has_think and has_boxed) else 0.0

    def extract_bounding_boxes(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        Extract bounding boxes from reasoning text for R_ground computation.
        
        Args:
            text: The reasoning text containing bounding box coordinates
            
        Returns:
            List of tuples (object_name, [x1, y1, x2, y2])
        """
        if not text:
            return []
        
        bboxes = []
        
        # Comprehensive patterns for extracting bounding boxes
        patterns = [
            # "I can see a cup located at [x1, y1, x2, y2]"
            r'(?:I\s+(?:can\s+see|notice|spot)\s+a\s+|There\'?s\s+a\s+)(\w+)\s+(?:located\s+|positioned\s+)?at\s+(?:coordinates\s+|bounding\s+box\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "cup is visible at [x1, y1, x2, y2]" 
            r'(\w+)\s+is\s+visible\s+at\s+\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "cup at bounding box [x1, y1, x2, y2]"
            r'(\w+)\s+at\s+(?:bounding\s+box\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "cup with center at [x, y]" (convert to bbox)
            r'(\w+)\s+with\s+center\s+at\s+\[(\d+\.?\d*),\s*(\d+\.?\d*)\]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 5:  # Has full bounding box coordinates
                    object_name = groups[0].lower()
                    coords = [int(float(groups[i])) for i in range(1, 5)]
                    bboxes.append((object_name, coords))
                elif len(groups) == 3:  # Center coordinates - convert to bbox
                    object_name = groups[0].lower()
                    center_x, center_y = float(groups[1]), float(groups[2])
                    # Assume a default size for bbox from center
                    half_size = 25  # pixels
                    coords = [
                        int(center_x - half_size), int(center_y - half_size),
                        int(center_x + half_size), int(center_y + half_size)
                    ]
                    bboxes.append((object_name, coords))
        
        return bboxes

    def extract_overlap_count(self, text: str) -> Optional[int]:
        """
        Extract the number of overlapping objects from reasoning text.
        
        Args:
            text: The reasoning text
            
        Returns:
            Number of overlapping objects, or None if not found
        """
        if not text:
            return None
            
        # Patterns to identify overlap mentions
        overlap_patterns = [
            r'(\d+)\s+(?:objects?\s+)?(?:are\s+)?(?:visible\s+in\s+)?both\s+(?:perspectives?|views?|images?)',
            r'(?:total\s+of\s+)?(\d+)\s+(?:unique\s+)?objects?\s+(?:appear\s+in\s+)?(?:more\s+than\s+one\s+view|multiple\s+views?)',
            r'(\d+)\s+overlapping\s+objects?',
            r'overlap\s+(?:count|number):\s*(\d+)',
            r'cross-view\s+(?:overlap|entities?):\s*(\d+)'
        ]
        
        for pattern in overlap_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
                    
        return None

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] of first box
            box2: [x1, y1, x2, y2] of second box
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Convert to float for calculations
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]
        
        # Calculate intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Check if there's no intersection
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area

    def hungarian_algorithm(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        Hungarian algorithm for optimal bipartite matching.
        
        Args:
            cost_matrix: 2D matrix where cost_matrix[i][j] is cost of assigning i to j
            
        Returns:
            List of (row, col) optimal assignments
        """
        if not cost_matrix or not cost_matrix[0]:
            return []
        
        rows = len(cost_matrix)
        cols = len(cost_matrix[0])
        
        # For simplicity, use greedy approach for larger matrices
        # Full Hungarian implementation would be more complex
        if rows <= 3 and cols <= 3:
            return self._brute_force_assignment(cost_matrix)
        else:
            return self._greedy_assignment(cost_matrix)

    def _brute_force_assignment(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """Brute force optimal assignment for small matrices."""
        import itertools
        
        rows = len(cost_matrix)
        cols = len(cost_matrix[0])
        
        best_cost = float('inf')
        best_assignment = []
        
        # Generate all possible assignments
        if rows <= cols:
            for perm in itertools.permutations(range(cols), rows):
                cost = sum(cost_matrix[i][perm[i]] for i in range(rows))
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = [(i, perm[i]) for i in range(rows)]
        else:
            for perm in itertools.permutations(range(rows), cols):
                cost = sum(cost_matrix[perm[j]][j] for j in range(cols))
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = [(perm[j], j) for j in range(cols)]
        
        return best_assignment

    def _greedy_assignment(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """Greedy assignment algorithm."""
        rows = len(cost_matrix)
        cols = len(cost_matrix[0])
        
        assignments = []
        used_rows = set()
        used_cols = set()
        
        # Create list of all (cost, row, col) tuples and sort by cost
        candidates = []
        for i in range(rows):
            for j in range(cols):
                candidates.append((cost_matrix[i][j], i, j))
        
        candidates.sort()  # Sort by cost (ascending)
        
        # Greedily select assignments
        for cost, row, col in candidates:
            if row not in used_rows and col not in used_cols:
                assignments.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return assignments

    def compute_r_ground(self, predicted_reasoning: str, ground_truth_reasoning: str) -> float:
        """
        Compute R_ground: Dense grounding reward based on IoU matching.
        
        Uses Hungarian algorithm to find optimal bipartite matching between
        predicted and ground truth bounding boxes, then computes average IoU.
        
        Args:
            predicted_reasoning: Model's reasoning text with bounding boxes
            ground_truth_reasoning: Ground truth reasoning text with bounding boxes
            
        Returns:
            Grounding reward between 0.0 and 1.0
        """
        if not predicted_reasoning or not ground_truth_reasoning:
            return 0.0
        
        # Extract bounding boxes
        pred_bboxes = self.extract_bounding_boxes(predicted_reasoning)
        gt_bboxes = self.extract_bounding_boxes(ground_truth_reasoning)
        
        if not pred_bboxes and not gt_bboxes:
            return 1.0  # Both have no bboxes, perfect match
        elif not pred_bboxes or not gt_bboxes:
            return 0.0  # One has bboxes, the other doesn't
        
        # Build cost matrix (negative IoU since Hungarian finds minimum cost)
        cost_matrix = []
        for pred_name, pred_box in pred_bboxes:
            row = []
            for gt_name, gt_box in gt_bboxes:
                # Only match boxes of same object type
                if pred_name.split('_')[0] == gt_name.split('_')[0]:
                    iou = self.calculate_iou(pred_box, gt_box)
                    row.append(-iou)  # Negative for minimization
                else:
                    row.append(-0.0)  # No match for different object types
            cost_matrix.append(row)
        
        if not cost_matrix:
            return 0.0
        
        # Find optimal assignment using Hungarian algorithm
        assignments = self.hungarian_algorithm(cost_matrix)
        
        # Calculate average IoU for optimal assignment
        total_iou = 0.0
        valid_assignments = 0
        
        for pred_idx, gt_idx in assignments:
            if pred_idx < len(pred_bboxes) and gt_idx < len(gt_bboxes):
                pred_name, pred_box = pred_bboxes[pred_idx]
                gt_name, gt_box = gt_bboxes[gt_idx]
                
                # Only count if object types match
                if pred_name.split('_')[0] == gt_name.split('_')[0]:
                    iou = self.calculate_iou(pred_box, gt_box)
                    total_iou += iou
                    valid_assignments += 1
        
        # Normalize by maximum possible assignments
        max_possible = max(len(pred_bboxes), len(gt_bboxes))
        return total_iou / max_possible if max_possible > 0 else 0.0

    def compute_r_overlap(self, predicted_reasoning: str, ground_truth_reasoning: str) -> float:
        """
        Compute R_overlap: Binary reward for cross-view overlap accuracy.
        
        Args:
            predicted_reasoning: Model's reasoning text
            ground_truth_reasoning: Ground truth reasoning text
            
        Returns:
            Binary reward (0.0 or 1.0)
        """
        pred_overlap = self.extract_overlap_count(predicted_reasoning)
        gt_overlap = self.extract_overlap_count(ground_truth_reasoning)
        
        # If both don't mention overlap, assume it's not required for this sample
        if pred_overlap is None and gt_overlap is None:
            return 1.0
        
        # If only one mentions overlap, it's a mismatch
        if pred_overlap is None or gt_overlap is None:
            return 0.0
        
        # Binary reward for exact match
        return 1.0 if pred_overlap == gt_overlap else 0.0

    def parse_answer(self, output_str: str) -> Optional[str]:
        """
        Parse the final answer from model output.
        
        Args:
            output_str: The model's output string
            
        Returns:
            Extracted answer or None if not found
        """
        # Extract answer from \\boxed{} format
        boxed_patterns = [
            r'\\\\boxed\{([^}]*)\}',  # \\boxed{answer}
            r'\\boxed\{([^}]*)\}',     # \boxed{answer}
            r'\$\\boxed\{([^}]*)\}\$', # $\boxed{answer}$
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, output_str)
            if match:
                return match.group(1).strip()
        
        return None

    def parse_coordinates(self, answer_str: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from answer string for grasping tasks.
        
        Args:
            answer_str: The answer string containing coordinates
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        if not answer_str:
            return None
        
        # Pattern for coordinate extraction: "image_number, [x, y]"
        coord_patterns = [
            r'[01],\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # "0, [x, y]" or "1, [x, y]"
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',          # Just "[x, y]"
            r'(\d+\.?\d*),\s*(\d+\.?\d*)'               # "x, y"
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, answer_str)
            if match:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    return (x, y)
                except (ValueError, IndexError):
                    continue
        
        return None

    def normalize_object_name(self, name: str) -> str:
        """
        Normalize an object name for comparison.
        
        Args:
            name: The object name to normalize
            
        Returns:
            Normalized object name string
        """
        if name is None:
            return ""
        
        # Convert to lowercase and strip whitespace
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'^(a\s+|an\s+|the\s+)', '', name)
        name = re.sub(r'(\s+object|\s+item)$', '', name)
        
        # Handle underscores and spaces consistently
        name = re.sub(r'[_\s]+', '_', name)
        
        # Remove trailing numbers/indices (e.g., "cup_1" -> "cup")
        name = re.sub(r'_\d+$', '', name)
        
        return name

    def compute_r_ans(self, predicted_answer: str, ground_truth_answer: str) -> float:
        """
        Compute R_ans: Task-specific answer correctness reward.
        
        For QA tasks: Binary reward (exact match)
        For grasping tasks: Distance-shaped reward
        
        Args:
            predicted_answer: Model's predicted answer
            ground_truth_answer: Ground truth answer
            
        Returns:
            Answer correctness reward between 0.0 and 1.0
        """
        if self.task_type == "grasping":
            # Parse coordinates for grasping task
            pred_coords = self.parse_coordinates(predicted_answer)
            gt_coords = self.parse_coordinates(ground_truth_answer)
            
            if pred_coords is None or gt_coords is None:
                return 0.0
            
            # Calculate Euclidean distance
            distance = math.sqrt(
                (pred_coords[0] - gt_coords[0])**2 + 
                (pred_coords[1] - gt_coords[1])**2
            )
            
            # Distance-shaped reward: max(0, 1 - distance/d_max)
            return max(0.0, 1.0 - distance / self.d_max)
        
        else:  # QA task
            # Normalize both answers for comparison
            pred_norm = self.normalize_object_name(predicted_answer)
            gt_norm = self.normalize_object_name(ground_truth_answer)
            
            # Binary reward for exact match
            return 1.0 if pred_norm == gt_norm else 0.0

    def parse_model_output(self, output_str: str) -> Dict[str, Optional[str]]:
        """
        Parse model output to extract reasoning and answer.
        
        Args:
            output_str: The raw model output string
            
        Returns:
            Dictionary with 'reasoning' and 'answer' fields
        """
        result = {
            'reasoning': None,
            'answer': None
        }
        
        # Extract reasoning from <think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, output_str, re.DOTALL)
        if think_match:
            result['reasoning'] = think_match.group(1).strip()
        
        # Extract answer
        result['answer'] = self.parse_answer(output_str)
        
        return result

    def compute_cvsr_reward(self, model_output: str, ground_truth: str) -> Dict[str, float]:
        """
        Compute the CVSR reward components.
        
        Args:
            model_output: Model's complete output
            ground_truth: Ground truth string
            
        Returns:
            Dictionary with CVSR component scores
        """
        # Parse outputs
        parsed_output = self.parse_model_output(model_output)
        parsed_gt = self.parse_model_output(ground_truth)
        
        predicted_reasoning = parsed_output['reasoning'] or ""
        predicted_answer = parsed_output['answer'] or ""
        
        gt_reasoning = parsed_gt['reasoning'] or ""
        gt_answer = parsed_gt['answer'] or ground_truth.strip()
        
        # Compute CVSR components
        r_ground = self.compute_r_ground(predicted_reasoning, gt_reasoning)
        r_overlap = self.compute_r_overlap(predicted_reasoning, gt_reasoning)
        r_ans = self.compute_r_ans(predicted_answer, gt_answer)
        
        # Compute weighted CVSR
        r_cvsr = (self.w_ground * r_ground + 
                  self.w_overlap * r_overlap + 
                  self.w_ans * r_ans)
        
        return {
            'r_ground': r_ground,
            'r_overlap': r_overlap,
            'r_ans': r_ans,
            'r_cvsr': r_cvsr
        }

    def compute_total_reward(self, model_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Compute the total CVSR reward: R = λ₁ * R_format + λ₂ * R_CVSR
        
        Args:
            model_output: Model's complete output
            ground_truth: Ground truth string
            
        Returns:
            Dictionary with detailed reward breakdown
        """
        # Compute format reward
        r_format = self.compute_r_format(model_output)
        
        # Compute CVSR components
        cvsr_results = self.compute_cvsr_reward(model_output, ground_truth)
        
        # Compute total reward
        total_reward = self.lambda1 * r_format + self.lambda2 * cvsr_results['r_cvsr']
        
        # Parse outputs for detailed info
        parsed_output = self.parse_model_output(model_output)
        parsed_gt = self.parse_model_output(ground_truth)
        
        return {
            'score': total_reward,
            'r_format': r_format,
            'r_cvsr': cvsr_results['r_cvsr'],
            'r_ground': cvsr_results['r_ground'],
            'r_overlap': cvsr_results['r_overlap'],
            'r_ans': cvsr_results['r_ans'],
            'predicted_answer': parsed_output['answer'],
            'ground_truth_answer': parsed_gt['answer'] or ground_truth.strip(),
            'has_reasoning': parsed_output['reasoning'] is not None,
            'weights': {
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'w_ground': self.w_ground,
                'w_overlap': self.w_overlap,
                'w_ans': self.w_ans
            },
            'task_type': self.task_type
        }


# Convenience functions for different task types
def get_cvsr_qa_reward(lambda1: float = 0.2, lambda2: float = 0.8,
                       w_ground: float = 0.3, w_overlap: float = 0.2, 
                       w_ans: float = 0.5) -> CVSRReward:
    """Get CVSR reward model configured for Q&A tasks."""
    return CVSRReward(
        lambda1=lambda1, lambda2=lambda2,
        w_ground=w_ground, w_overlap=w_overlap, w_ans=w_ans,
        task_type="qa"
    )


def get_cvsr_grasping_reward(lambda1: float = 0.2, lambda2: float = 0.8,
                            w_ground: float = 0.3, w_overlap: float = 0.2,
                            w_ans: float = 0.5, d_max: float = 100.0) -> CVSRReward:
    """Get CVSR reward model configured for grasping tasks."""
    return CVSRReward(
        lambda1=lambda1, lambda2=lambda2,
        w_ground=w_ground, w_overlap=w_overlap, w_ans=w_ans,
        d_max=d_max, task_type="grasping"
    )


def compute_cvsr_reward(model_output: str, ground_truth: str, 
                       task_type: str = "qa", **kwargs) -> float:
    """
    Convenience function to compute CVSR reward directly.
    
    Args:
        model_output: Model's output string
        ground_truth: Ground truth string
        task_type: "qa" or "grasping"
        **kwargs: Additional parameters for CVSRReward
        
    Returns:
        Total reward score between 0.0 and 1.0
    """
    if task_type == "grasping":
        reward_model = get_cvsr_grasping_reward(**kwargs)
    else:
        reward_model = get_cvsr_qa_reward(**kwargs)
    
    result = reward_model.compute_total_reward(model_output, ground_truth)
    return result['score']


if __name__ == "__main__":
    # Test cases for CVSR reward
    print("Testing Cross-View Spatial Reward (CVSR)")
    print("=" * 60)
    
    # Test case 1: Perfect format and content (QA task)
    test1_output = """
    <think>
    I need to analyze images from two robot perspectives to answer the spatial reasoning question.
    
    I can see a cup located at [100, 150, 200, 250] in the main perspective.
    I notice a tomato at [300, 100, 400, 200] in the auxiliary perspective.
    I spot a bowl at [500, 300, 600, 400] visible in both perspectives.
    
    Cross-view overlap: 1 unique object appears in more than one view (the bowl).
    
    Distance from cup to tomato: 223.6 pixels
    Distance from cup to bowl: 458.3 pixels
    
    Therefore, bowl is farthest from cup.
    </think>
    
    \\boxed{bowl}
    """
    
    test1_gt = """
    <think>
    Analyzing spatial relationships from dual robot perspectives.
    
    Cup at bounding box [105, 155, 205, 255].
    Tomato is visible at [295, 95, 395, 195].
    Bowl at [495, 295, 595, 395].
    
    Cross-view entities: 1 object appears in multiple views.
    
    Distance calculations show bowl is farthest from cup.
    </think>
    
    \\boxed{bowl}
    """
    
    # Test QA task
    cvsr_qa = get_cvsr_qa_reward()
    result1 = cvsr_qa.compute_total_reward(test1_output, test1_gt)
    
    print("Test 1 - QA Task (Perfect Match):")
    print(f"  Total Score: {result1['score']:.3f}")
    print(f"  R_format: {result1['r_format']:.3f}")
    print(f"  R_CVSR: {result1['r_cvsr']:.3f}")
    print(f"    - R_ground: {result1['r_ground']:.3f}")
    print(f"    - R_overlap: {result1['r_overlap']:.3f}")
    print(f"    - R_ans: {result1['r_ans']:.3f}")
    print(f"  Predicted: {result1['predicted_answer']}")
    print(f"  Ground Truth: {result1['ground_truth_answer']}")
    print()
    
    # Test case 2: Grasping task
    test2_output = """
    <think>
    Analyzing dual robot perspectives for grasping task.
    
    I can see a cup at [100, 100, 200, 200] in main perspective.
    There's a bowl positioned at [300, 300, 400, 400] in auxiliary perspective.
    
    Cross-view overlap: 0 objects appear in both views.
    
    Target object is in main perspective (image 0).
    Center coordinates: [150.0, 150.0]
    </think>
    
    \\boxed{0, [150.0, 150.0]}
    """
    
    test2_gt = """
    <think>
    Grasping analysis from dual perspectives.
    
    Cup located at [95, 95, 205, 205] in main view.
    Bowl at [295, 295, 405, 405] in auxiliary view.
    
    Target for grasping at center [150.0, 150.0].
    </think>
    
    \\boxed{0, [152.0, 148.0]}
    """
    
    # Test grasping task
    cvsr_grasping = get_cvsr_grasping_reward(d_max=50.0)
    result2 = cvsr_grasping.compute_total_reward(test2_output, test2_gt)
    
    print("Test 2 - Grasping Task:")
    print(f"  Total Score: {result2['score']:.3f}")
    print(f"  R_format: {result2['r_format']:.3f}")
    print(f"  R_CVSR: {result2['r_cvsr']:.3f}")
    print(f"    - R_ground: {result2['r_ground']:.3f}")
    print(f"    - R_overlap: {result2['r_overlap']:.3f}")
    print(f"    - R_ans: {result2['r_ans']:.3f}")
    print(f"  Coordinate Distance: {math.sqrt((150.0-152.0)**2 + (150.0-148.0)**2):.3f}")
    print()
    
    # Test case 3: Poor format (missing think tags)
    test3_output = "\\boxed{cup}"
    test3_gt = "\\boxed{bowl}"
    
    result3 = cvsr_qa.compute_total_reward(test3_output, test3_gt)
    
    print("Test 3 - Poor Format (No Think Tags):")
    print(f"  Total Score: {result3['score']:.3f}")
    print(f"  R_format: {result3['r_format']:.3f}")
    print(f"  R_CVSR: {result3['r_cvsr']:.3f}")
    print()
    
    # Test case 4: Different weight configurations
    print("Test 4 - Different Weight Configurations:")
    print("-" * 40)
    
    weight_configs = [
        {"lambda1": 0.5, "lambda2": 0.5, "name": "Equal λ weights"},
        {"lambda1": 0.1, "lambda2": 0.9, "name": "CVSR-heavy"},
        {"w_ground": 0.6, "w_overlap": 0.2, "w_ans": 0.2, "name": "Grounding-heavy"},
        {"w_ground": 0.1, "w_overlap": 0.1, "w_ans": 0.8, "name": "Answer-heavy"}
    ]
    
    for config in weight_configs:
        name = config.pop("name")
        test_model = get_cvsr_qa_reward(**config)
        result = test_model.compute_total_reward(test1_output, test1_gt)
        print(f"  {name}: Score = {result['score']:.3f}")
    
    print()
    print("CVSR Implementation Complete!")