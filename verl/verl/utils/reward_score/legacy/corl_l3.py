"""
Reward function for CORL Level 3 dual-robot perspective grasping tasks
"""

import re
from typing import Union, Dict, Any, Optional, List, Tuple
import json
import math


class CorlL3GraspingReward:
    """
    Reward model for evaluating Level 3 grasping responses from dual robot perspectives.
    
    The model expects responses in the following format:
    <think>
    [reasoning process]
    </think>
    
    \boxed{image_number, [x, y]}
    
    Where image_number is 0 (main perspective) or 1 (auxiliary perspective),
    and [x, y] are the center coordinates of the target object.
    """
    
    def __init__(self, partial_credit: bool = False, bbox_reward_weight: float = 0.3, 
                 coordinate_threshold: float = 50.0):
        """
        Initialize the reward model.
        
        Args:
            partial_credit: Whether to give partial credit for partially correct answers
            bbox_reward_weight: Weight for bounding box IoU reward (0.0-1.0)
            coordinate_threshold: Distance threshold for coordinate accuracy (pixels)
        """
        self.partial_credit = partial_credit
        self.bbox_reward_weight = bbox_reward_weight
        self.answer_reward_weight = 1.0 - bbox_reward_weight
        self.coordinate_threshold = coordinate_threshold
        
    def parse_model_output(self, output_str: str) -> Dict[str, Optional[str]]:
        """
        Parse the model output to extract the reasoning and final answer.
        
        Args:
            output_str: The raw model output string
            
        Returns:
            Dictionary containing 'reasoning', 'answer', 'image_number', and 'coordinates' fields
        """
        result = {
            'reasoning': None,
            'answer': None,
            'image_number': None,
            'coordinates': None
        }
        
        # Extract reasoning from <think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, output_str, re.DOTALL)
        if think_match:
            result['reasoning'] = think_match.group(1).strip()
        
        # Extract answer from \boxed{} format - Level 3 specific: image_number, [x, y]
        boxed_patterns = [
            r'\\\\boxed\{([^}]*)\}',  # \\boxed{answer}
            r'\\boxed\{([^}]*)\}',     # \boxed{answer}
            r'\$\\boxed\{([^}]*)\}\$', # $\boxed{answer}$
        ]
        
        for pattern in boxed_patterns:
            boxed_match = re.search(pattern, output_str)
            if boxed_match:
                extracted_answer = boxed_match.group(1).strip()
                
                # Parse Level 3 format: image_number, [x, y]
                if self._parse_l3_answer_format(extracted_answer, result):
                    result['answer'] = extracted_answer
                    break
        
        # Fallback: try to find coordinates pattern directly
        if result['answer'] is None:
            coord_pattern = r'(\d+),?\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
            coord_match = re.search(coord_pattern, output_str)
            if coord_match:
                image_num = int(coord_match.group(1))
                x_coord = float(coord_match.group(2))
                y_coord = float(coord_match.group(3))
                
                if image_num in [0, 1]:
                    result['image_number'] = image_num
                    result['coordinates'] = [x_coord, y_coord]
                    result['answer'] = f"{image_num}, [{x_coord}, {y_coord}]"
        
        return result
    
    def _parse_l3_answer_format(self, answer: str, result_dict: Dict) -> bool:
        """
        Parse Level 3 answer format: image_number, [x, y]
        
        Args:
            answer: The extracted answer string
            result_dict: Dictionary to store parsed results
            
        Returns:
            True if parsing successful, False otherwise
        """
        if not answer or len(answer.strip()) == 0:
            return False
        
        answer = answer.strip()
        
        # Pattern to match: number, [number, number]
        # Allow for various whitespace and decimal numbers
        pattern = r'(\d+),?\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        match = re.match(pattern, answer)
        
        if match:
            image_num = int(match.group(1))
            x_coord = float(match.group(2))
            y_coord = float(match.group(3))
            
            # Validate image number (should be 0 or 1)
            if image_num not in [0, 1]:
                return False
            
            # Validate coordinates (should be reasonable pixel values)
            if x_coord < 0 or y_coord < 0 or x_coord > 2000 or y_coord > 2000:
                return False
            
            result_dict['image_number'] = image_num
            result_dict['coordinates'] = [x_coord, y_coord]
            return True
        
        return False
    
    def extract_bounding_boxes(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        Extract bounding boxes from reasoning text.
        
        Args:
            text: The reasoning text containing bounding box coordinates
            
        Returns:
            List of tuples (object_name, [x1, y1, x2, y2])
        """
        if not text:
            return []
        
        bboxes = []
        
        # Pattern to match object mentions with bounding boxes
        pattern = r'(?:I\s+(?:can\s+see|notice|spot)\s+a\s+|There\'?s\s+a\s+|Additionally,?\s+there\'?s\s+(?:a\s+|another\s+)?|I\s+also\s+notice\s+a\s+(?:second\s+|third\s+)?)?(\w+)\s+(?:located\s+|positioned\s+|is\s+visible\s+)?at\s+(?:coordinates\s+|bounding\s+box\s+|position\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            object_name = groups[0].lower()
            coords = [int(groups[i]) for i in range(1, 5)]
            bboxes.append((object_name, coords))
        
        return bboxes
    
    def calculate_distance(self, coord1: List[float], coord2: List[float]) -> float:
        """
        Calculate Euclidean distance between two coordinate points.
        
        Args:
            coord1: [x, y] coordinates of first point
            coord2: [x, y] coordinates of second point
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
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
    
    def compute_bbox_reward(self, predicted_reasoning: str, ground_truth_reasoning: str) -> float:
        """
        Compute reward based on bounding box IoU matching between predicted and ground truth reasoning.
        
        Args:
            predicted_reasoning: The model's reasoning text
            ground_truth_reasoning: The ground truth reasoning text
            
        Returns:
            Bounding box reward score between 0.0 and 1.0
        """
        if not predicted_reasoning or not ground_truth_reasoning:
            return 0.0
        
        # Extract bounding boxes from both texts
        pred_bboxes = self.extract_bounding_boxes(predicted_reasoning)
        gt_bboxes = self.extract_bounding_boxes(ground_truth_reasoning)
        
        if not pred_bboxes and not gt_bboxes:
            return 1.0  # Both have no bboxes, perfect match
        elif not pred_bboxes or not gt_bboxes:
            return 0.0  # One has bboxes, the other doesn't
        
        # Group by object type
        pred_by_type = {}
        for obj_name, coords in pred_bboxes:
            obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
            if obj_type not in pred_by_type:
                pred_by_type[obj_type] = []
            pred_by_type[obj_type].append(coords)
        
        gt_by_type = {}
        for obj_name, coords in gt_bboxes:
            obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
            if obj_type not in gt_by_type:
                gt_by_type[obj_type] = []
            gt_by_type[obj_type].append(coords)
        
        # Calculate IoU for each object type using Hungarian algorithm
        total_ious = []
        all_object_types = set(pred_by_type.keys()) | set(gt_by_type.keys())
        
        for obj_type in all_object_types:
            pred_boxes = pred_by_type.get(obj_type, [])
            gt_boxes = gt_by_type.get(obj_type, [])
            
            if not pred_boxes or not gt_boxes:
                total_ious.append(0.0)  # Missing predictions or ground truth
                continue
            
            # Use Hungarian algorithm for optimal assignment
            type_iou = self.optimal_bbox_matching(pred_boxes, gt_boxes)
            total_ious.append(type_iou)
        
        # Return overall average IoU
        return sum(total_ious) / len(total_ious) if total_ious else 0.0
    
    def optimal_bbox_matching(self, pred_boxes: List[List[int]], gt_boxes: List[List[int]]) -> float:
        """
        Find optimal matching between predicted and ground truth boxes using Hungarian algorithm.
        
        Args:
            pred_boxes: List of predicted bounding boxes
            gt_boxes: List of ground truth bounding boxes
            
        Returns:
            Average IoU of optimal matching
        """
        if not pred_boxes or not gt_boxes:
            return 0.0
        
        # Build cost matrix (we use negative IoU since Hungarian finds minimum cost)
        cost_matrix = []
        for pred_box in pred_boxes:
            row = []
            for gt_box in gt_boxes:
                iou = self.calculate_iou(pred_box, gt_box)
                row.append(-iou)  # Negative because Hungarian finds minimum
            cost_matrix.append(row)
        
        # Apply Hungarian algorithm for optimal assignment
        assignments = self.hungarian_algorithm(cost_matrix)
        
        # Calculate average IoU for the optimal assignment
        total_iou = 0.0
        valid_assignments = 0
        
        for pred_idx, gt_idx in assignments:
            if pred_idx < len(pred_boxes) and gt_idx < len(gt_boxes):
                iou = self.calculate_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
                total_iou += iou
                valid_assignments += 1
        
        # Handle case where we have different numbers of boxes
        max_possible_matches = max(len(pred_boxes), len(gt_boxes))
        if max_possible_matches > 0:
            # Penalize for having different numbers of boxes
            return total_iou / max_possible_matches
        else:
            return 0.0
    
    def hungarian_algorithm(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        Simple implementation of Hungarian algorithm for optimal assignment.
        
        Args:
            cost_matrix: 2D matrix where cost_matrix[i][j] is cost of assigning i to j
            
        Returns:
            List of (row, col) assignments
        """
        if not cost_matrix or not cost_matrix[0]:
            return []
        
        rows = len(cost_matrix)
        cols = len(cost_matrix[0])
        
        # For small matrices, use brute force (more reliable)
        if rows <= 3 and cols <= 3:
            return self.brute_force_assignment(cost_matrix)
        
        # For larger matrices, use a simplified greedy approach
        # (Full Hungarian algorithm would be more complex to implement correctly)
        return self.greedy_assignment(cost_matrix)
    
    def brute_force_assignment(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        Brute force optimal assignment for small matrices.
        """
        import itertools
        
        rows = len(cost_matrix)
        cols = len(cost_matrix[0])
        
        best_cost = float('inf')
        best_assignment = []
        
        # Generate all possible assignments
        if rows <= cols:
            # Assign each row to a column
            for perm in itertools.permutations(range(cols), rows):
                cost = sum(cost_matrix[i][perm[i]] for i in range(rows))
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = [(i, perm[i]) for i in range(rows)]
        else:
            # Assign each column to a row
            for perm in itertools.permutations(range(rows), cols):
                cost = sum(cost_matrix[perm[j]][j] for j in range(cols))
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = [(perm[j], j) for j in range(cols)]
        
        return best_assignment
    
    def greedy_assignment(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        Greedy assignment algorithm (approximation to Hungarian).
        """
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
    
    def compute_coordinate_reward(self, pred_image_num: int, pred_coords: List[float], 
                                gt_image_num: int, gt_coords: List[float]) -> float:
        """
        Compute reward based on image number and coordinate accuracy.
        
        Args:
            pred_image_num: Predicted image number (0 or 1)
            pred_coords: Predicted [x, y] coordinates
            gt_image_num: Ground truth image number (0 or 1)
            gt_coords: Ground truth [x, y] coordinates
            
        Returns:
            Coordinate reward score between 0.0 and 1.0
        """
        # Check image number first
        if pred_image_num != gt_image_num:
            return 0.0  # Wrong image, no points
        
        # Calculate coordinate distance
        distance = self.calculate_distance(pred_coords, gt_coords)
        
        # Apply distance-based scoring
        if distance <= self.coordinate_threshold:
            # Linear decay from 1.0 at distance 0 to 0.5 at threshold
            score = 1.0 - (distance / self.coordinate_threshold) * 0.5
            return max(0.5, score)
        else:
            # Exponential decay beyond threshold
            decay_factor = (distance - self.coordinate_threshold) / self.coordinate_threshold
            score = 0.5 * math.exp(-decay_factor)
            return max(0.0, score)
    
    def reward(self, model_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Main reward function that evaluates model output against ground truth.
        
        Args:
            model_output: The complete model output string
            ground_truth: The ground truth string (includes reasoning and answer)
            
        Returns:
            Dictionary containing:
                - 'score': The reward score (0.0 to 1.0)
                - 'predicted_answer': The extracted answer from model output
                - 'ground_truth_answer': The extracted answer from ground truth
                - 'has_reasoning': Whether the model included reasoning
                - 'bbox_score': The bounding box IoU score
                - 'coordinate_score': The coordinate accuracy score
                - 'image_number_correct': Whether image number is correct
                - 'details': Additional evaluation details
        """
        # Parse model output
        parsed_output = self.parse_model_output(model_output)
        predicted_answer = parsed_output['answer']
        predicted_reasoning = parsed_output['reasoning']
        pred_image_num = parsed_output['image_number']
        pred_coords = parsed_output['coordinates']
        
        # Parse ground truth
        parsed_gt = self.parse_model_output(ground_truth)
        gt_answer = parsed_gt['answer']
        gt_reasoning = parsed_gt['reasoning']
        gt_image_num = parsed_gt['image_number']
        gt_coords = parsed_gt['coordinates']
        
        # If ground truth doesn't have structured format, treat it as plain answer
        if gt_answer is None:
            gt_answer = ground_truth.strip()
        
        # Check if prediction format is valid
        if (predicted_answer is None or pred_image_num is None or pred_coords is None or
            gt_image_num is None or gt_coords is None):
            # Invalid format - return 0 score
            result = {
                'score': 0.0,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': gt_answer,
                'has_reasoning': predicted_reasoning is not None,
                'bbox_score': 0.0,
                'coordinate_score': 0.0,
                'image_number_correct': False,
                'details': {
                    'predicted_image_number': pred_image_num,
                    'predicted_coordinates': pred_coords,
                    'ground_truth_image_number': gt_image_num,
                    'ground_truth_coordinates': gt_coords,
                    'coordinate_distance': None,
                    'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                    'bbox_reward_weight': self.bbox_reward_weight,
                    'coordinate_reward_weight': self.answer_reward_weight,
                    'predicted_bboxes': [],
                    'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                    'invalid_format': True
                }
            }
            return result
        
        # Compute coordinate reward score (main component for Level 3)
        coordinate_score = self.compute_coordinate_reward(
            pred_image_num, pred_coords, gt_image_num, gt_coords
        )
        
        # Compute bounding box reward score (secondary component)
        bbox_score = 0.0
        if self.bbox_reward_weight > 0 and predicted_reasoning and gt_reasoning:
            bbox_score = self.compute_bbox_reward(predicted_reasoning, gt_reasoning)
        
        # Check image number correctness
        image_number_correct = (pred_image_num == gt_image_num)
        
        # Compute weighted final score
        if self.bbox_reward_weight > 0:
            final_score = (self.answer_reward_weight * coordinate_score + 
                          self.bbox_reward_weight * bbox_score)
        else:
            final_score = coordinate_score
        
        # Calculate coordinate distance for details
        coord_distance = self.calculate_distance(pred_coords, gt_coords)
        
        # Prepare result
        result = {
            'score': final_score,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': gt_answer,
            'has_reasoning': predicted_reasoning is not None,
            'bbox_score': bbox_score,
            'coordinate_score': coordinate_score,
            'image_number_correct': image_number_correct,
            'details': {
                'predicted_image_number': pred_image_num,
                'predicted_coordinates': pred_coords,
                'ground_truth_image_number': gt_image_num,
                'ground_truth_coordinates': gt_coords,
                'coordinate_distance': coord_distance,
                'coordinate_threshold': self.coordinate_threshold,
                'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                'bbox_reward_weight': self.bbox_reward_weight,
                'coordinate_reward_weight': self.answer_reward_weight,
                'predicted_bboxes': self.extract_bounding_boxes(predicted_reasoning or ""),
                'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                'invalid_format': False
            }
        }
        
        return result


def get_reward_model(config: Optional[Dict[str, Any]] = None) -> CorlL3GraspingReward:
    """
    Factory function to get the reward model instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized reward model
    """
    if config is None:
        config = {}
    
    return CorlL3GraspingReward(
        partial_credit=config.get('partial_credit', False),
        bbox_reward_weight=config.get('bbox_reward_weight', 0.3),
        coordinate_threshold=config.get('coordinate_threshold', 50.0)
    )


# Convenience function for direct scoring
def compute_reward(model_output: str, ground_truth: str, partial_credit: bool = False, 
                   bbox_reward_weight: float = 0.3, coordinate_threshold: float = 50.0) -> float:
    """
    Convenience function to compute reward directly.
    
    Args:
        model_output: The model's output
        ground_truth: The ground truth
        partial_credit: Whether to enable partial credit
        bbox_reward_weight: Weight for bounding box IoU reward (0.0-1.0)
        coordinate_threshold: Distance threshold for coordinate accuracy
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    reward_model = CorlL3GraspingReward(
        partial_credit=partial_credit, 
        bbox_reward_weight=bbox_reward_weight,
        coordinate_threshold=coordinate_threshold
    )
    result = reward_model.reward(model_output, ground_truth)
    return result['score']


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute score function expected by the reward system.
    
    Args:
        solution_str: The model's output/solution
        ground_truth: The ground truth answer
        
    Returns:
        Score between 0.0 and 1.0
    """
    return compute_reward(solution_str, ground_truth, partial_credit=True, bbox_reward_weight=0.3)


if __name__ == "__main__":
    # Test cases
    print("Testing CORL Level 3 Grasping Reward Function")
    print("=" * 50)
    
    # Initialize reward model
    reward_model = CorlL3GraspingReward(partial_credit=True, bbox_reward_weight=0.3, coordinate_threshold=50.0)
    
    # Test case 1: Correct answer with reasoning
    test1_output = """
    <think>
    Looking at both robot perspectives, I need to identify the target object for grasping.
    I can see a cup at bounding box [100, 150, 200, 250] in the main perspective.
    The target object is clearly visible in image 0 (main perspective).
    The center coordinates of the cup are [150, 200].
    </think>
    
    \\boxed{0, [150.0, 200.0]}
    """
    test1_gt = """
    <think>
    I can see a cup at bounding box [100, 150, 200, 250] in the main perspective.
    The center coordinates are [150, 200].
    </think>
    
    \\boxed{0, [150.0, 200.0]}
    """
    
    result1 = reward_model.reward(test1_output, test1_gt)
    print(f"Test 1 - Perfect match:")
    print(f"  Score: {result1['score']:.3f}")
    print(f"  Predicted: {result1['predicted_answer']}")
    print(f"  Ground Truth: {result1['ground_truth_answer']}")
    print(f"  Image Number Correct: {result1['image_number_correct']}")
    print(f"  Coordinate Distance: {result1['details']['coordinate_distance']:.2f}")
    print()
    
    # Test case 2: Correct image, slightly off coordinates
    test2_output = "\\boxed{0, [155.0, 205.0]}"
    test2_gt = "\\boxed{0, [150.0, 200.0]}"
    
    result2 = reward_model.reward(test2_output, test2_gt)
    print(f"Test 2 - Close coordinates:")
    print(f"  Score: {result2['score']:.3f}")
    print(f"  Coordinate Distance: {result2['details']['coordinate_distance']:.2f}")
    print(f"  Coordinate Score: {result2['coordinate_score']:.3f}")
    print()
    
    # Test case 3: Wrong image number
    test3_output = "\\boxed{1, [150.0, 200.0]}"
    test3_gt = "\\boxed{0, [150.0, 200.0]}"
    
    result3 = reward_model.reward(test3_output, test3_gt)
    print(f"Test 3 - Wrong image number:")
    print(f"  Score: {result3['score']:.3f}")
    print(f"  Image Number Correct: {result3['image_number_correct']}")
    print()
    
    # Test case 4: Far coordinates
    test4_output = "\\boxed{0, [250.0, 300.0]}"
    test4_gt = "\\boxed{0, [150.0, 200.0]}"
    
    result4 = reward_model.reward(test4_output, test4_gt)
    print(f"Test 4 - Far coordinates:")
    print(f"  Score: {result4['score']:.3f}")
    print(f"  Coordinate Distance: {result4['details']['coordinate_distance']:.2f}")
    print(f"  Coordinate Score: {result4['coordinate_score']:.3f}")
    print()
    
    # Test case 5: Invalid format
    test5_output = "The answer is somewhere on the left"
    test5_gt = "\\boxed{0, [150.0, 200.0]}"
    
    result5 = reward_model.reward(test5_output, test5_gt)
    print(f"Test 5 - Invalid format:")
    print(f"  Score: {result5['score']:.3f}")
    print(f"  Invalid Format: {result5['details']['invalid_format']}")
    print()
    
    # Test distance thresholds
    print("=" * 50)
    print("Testing Distance Thresholds:")
    print("=" * 50)
    
    distances_to_test = [0, 10, 25, 50, 75, 100, 150, 200]
    for dist in distances_to_test:
        test_output = f"\\boxed{{0, [{150 + dist}, 200.0]}}"
        test_gt = "\\boxed{0, [150.0, 200.0]}"
        
        result = reward_model.reward(test_output, test_gt)
        print(f"Distance {dist:3d}: Score = {result['coordinate_score']:.3f}")