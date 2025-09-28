"""
Reward function for CORL Level 2 dual-robot perspective spatial understanding tasks
"""

import re
from typing import Union, Dict, Any, Optional, List, Tuple
import json


class CorlL2SpatialReward:
    """
    Reward model for evaluating Level 2 spatial understanding responses from dual robot perspectives.
    
    The model expects responses in the following format:
    <think>
    [reasoning process with bounding box coordinates]
    </think>
    
    \boxed{object_name}
    
    The tasks involve spatial reasoning questions like:
    - "Which object is farthest from X?"
    - "Which object is closest to Y?"
    - Other spatial relationship queries
    """
    
    def __init__(self, partial_credit: bool = False, bbox_reward_weight: float = 0.3):
        """
        Initialize the reward model.
        
        Args:
            partial_credit: Whether to give partial credit for partially correct answers
            bbox_reward_weight: Weight for bounding box IoU reward (0.0-1.0)
        """
        self.partial_credit = partial_credit
        self.bbox_reward_weight = bbox_reward_weight
        self.answer_reward_weight = 1.0 - bbox_reward_weight
        
    def parse_model_output(self, output_str: str) -> Dict[str, Optional[str]]:
        """
        Parse the model output to extract the reasoning and final answer.
        
        Args:
            output_str: The raw model output string
            
        Returns:
            Dictionary containing 'reasoning' and 'answer' fields
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
        
        # Extract answer from \boxed{} format
        boxed_patterns = [
            r'\\\\boxed\{([^}]*)\}',  # \\boxed{answer}
            r'\\boxed\{([^}]*)\}',     # \boxed{answer}
            r'\$\\boxed\{([^}]*)\}\$', # $\boxed{answer}$
        ]
        
        for pattern in boxed_patterns:
            boxed_match = re.search(pattern, output_str)
            if boxed_match:
                extracted_answer = boxed_match.group(1).strip()
                
                # Validate that the answer is a reasonable object name format
                if self._is_valid_object_name(extracted_answer):
                    result['answer'] = extracted_answer.lower()
                    break
        
        # Fallback: try to find answer after "Answer:" or similar keywords
        if result['answer'] is None:
            answer_pattern = r'(?:Answer|Final answer|The answer is)[:\s]+([^\n.]+)'
            answer_match = re.search(answer_pattern, output_str, re.IGNORECASE)
            if answer_match:
                candidate = answer_match.group(1).strip()
                if self._is_valid_object_name(candidate):
                    result['answer'] = candidate.lower()
        
        return result
    
    def _is_valid_object_name(self, answer: str) -> bool:
        """
        Validate if the extracted answer is a reasonable object name.
        
        Args:
            answer: The extracted answer string
            
        Returns:
            True if the answer format is valid, False otherwise
        """
        if not answer or len(answer.strip()) == 0:
            return False
        
        answer = answer.strip().lower()
        
        # Check for obvious invalid formats
        # 1. Should not contain coordinate patterns
        if re.search(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', answer):
            return False
        
        # 2. Should not contain excessive numbers (likely coordinates)
        number_count = len(re.findall(r'\d+', answer))
        if number_count > 2:  # Allow some numbers in object names like "cup_1"
            return False
        
        # 3. Should not be too long (reasonable object name length)
        if len(answer) > 50:
            return False
        
        # 4. Should not contain excessive whitespace or special characters
        if answer.count('\n') > 0 or len(answer.split()) > 5:
            return False
        
        # 5. Should not start with articles or common question words
        invalid_starts = ['the', 'a', 'an', 'what', 'which', 'how', 'where', 'when']
        first_word = answer.split()[0] if answer.split() else ''
        if first_word in invalid_starts:
            return False
        
        return True
    
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
        # More flexible pattern for Level 2 spatial reasoning text
        patterns = [
            # "I can see a cup located at [x1, y1, x2, y2]"
            r'(?:I\s+(?:can\s+see|notice|spot)\s+a\s+|There\'?s\s+a\s+)(\w+)\s+(?:located\s+|positioned\s+)?at\s+(?:coordinates\s+|bounding\s+box\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "cup is visible at [x1, y1, x2, y2]"
            r'(\w+)\s+is\s+visible\s+at\s+\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "Distance from reference_obj to target_obj: ..." (extract objects mentioned in distance calculations)
            r'Distance\s+from\s+(\w+)\s+to\s+(\w+):',
            # "cup at bounding box [x1, y1, x2, y2]"
            r'(\w+)\s+at\s+(?:bounding\s+box\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 5:  # Has coordinates
                    object_name = groups[0].lower()
                    coords = [int(groups[i]) for i in range(1, 5)]
                    bboxes.append((object_name, coords))
                elif len(groups) == 2:  # Distance pattern - just extract object names
                    # For distance patterns, we don't have coordinates but know objects exist
                    obj1, obj2 = groups[0].lower(), groups[1].lower()
                    # Add placeholder bboxes (will be handled in IoU calculation)
                    if not any(obj1 == name for name, _ in bboxes):
                        bboxes.append((obj1, [0, 0, 1, 1]))  # Placeholder
                    if not any(obj2 == name for name, _ in bboxes):
                        bboxes.append((obj2, [0, 0, 1, 1]))  # Placeholder
        
        return bboxes
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] of first box
            box2: [x1, y1, x2, y2] of second box
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Handle placeholder boxes (return 1.0 if both are placeholders, 0.0 if only one is)
        is_placeholder_1 = (box1 == [0, 0, 1, 1])
        is_placeholder_2 = (box2 == [0, 0, 1, 1])
        
        if is_placeholder_1 and is_placeholder_2:
            return 1.0  # Both are placeholders, consider them matching
        elif is_placeholder_1 or is_placeholder_2:
            return 0.0  # One is placeholder, one is real
        
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
        
        # Group by object type for matching
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
    
    def compute_answer_reward(self, predicted_answer: str, ground_truth: str) -> float:
        """
        Compute the reward score for a predicted answer.
        
        Args:
            predicted_answer: The model's predicted answer
            ground_truth: The correct answer
            
        Returns:
            Answer reward score between 0.0 and 1.0
        """
        # Normalize both answers
        pred_norm = self.normalize_object_name(predicted_answer)
        gt_norm = self.normalize_object_name(ground_truth)
        
        # Check for exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # If partial credit is enabled, check for partial matches
        if self.partial_credit:
            # Check if the ground truth is contained in the prediction or vice versa
            if gt_norm and (gt_norm in pred_norm or pred_norm in gt_norm):
                return 0.7
            
            # Check for similar object types (e.g., "cup" vs "mug")
            similar_pairs = [
                ('cup', 'mug', 'glass'),
                ('bowl', 'plate', 'dish'),
                ('bottle', 'container'),
                ('box', 'cube', 'container'),
                ('ball', 'sphere'),
            ]
            
            for similar_group in similar_pairs:
                if (pred_norm in similar_group and gt_norm in similar_group and 
                    pred_norm != gt_norm):
                    return 0.5
        
        return 0.0
    
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
                - 'answer_score': The final answer accuracy score
                - 'details': Additional evaluation details
        """
        # Parse model output
        parsed_output = self.parse_model_output(model_output)
        predicted_answer = parsed_output['answer']
        predicted_reasoning = parsed_output['reasoning']
        
        # Parse ground truth
        parsed_gt = self.parse_model_output(ground_truth)
        gt_answer = parsed_gt['answer']
        gt_reasoning = parsed_gt['reasoning']
        
        # If ground truth doesn't have structured format, treat it as plain answer
        if gt_answer is None:
            gt_answer = ground_truth.strip()
        
        # Check if answer format is valid
        if predicted_answer is None:
            # Invalid format - return 0 score
            result = {
                'score': 0.0,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': gt_answer,
                'has_reasoning': predicted_reasoning is not None,
                'bbox_score': 0.0,
                'answer_score': 0.0,
                'details': {
                    'normalized_prediction': None,
                    'normalized_ground_truth': self.normalize_object_name(gt_answer),
                    'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                    'bbox_reward_weight': self.bbox_reward_weight,
                    'answer_reward_weight': self.answer_reward_weight,
                    'predicted_bboxes': [],
                    'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                    'invalid_format': True
                }
            }
            return result
        
        # Compute answer reward score (main component for Level 2)
        answer_score = self.compute_answer_reward(predicted_answer, gt_answer)
        
        # Compute bounding box reward score (secondary component)
        bbox_score = 0.0
        if self.bbox_reward_weight > 0 and predicted_reasoning and gt_reasoning:
            bbox_score = self.compute_bbox_reward(predicted_reasoning, gt_reasoning)
        
        # Compute weighted final score
        if self.bbox_reward_weight > 0:
            final_score = (self.answer_reward_weight * answer_score + 
                          self.bbox_reward_weight * bbox_score)
        else:
            final_score = answer_score
        
        # Prepare result
        result = {
            'score': final_score,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': gt_answer,
            'has_reasoning': predicted_reasoning is not None,
            'bbox_score': bbox_score,
            'answer_score': answer_score,
            'details': {
                'normalized_prediction': self.normalize_object_name(predicted_answer),
                'normalized_ground_truth': self.normalize_object_name(gt_answer),
                'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                'bbox_reward_weight': self.bbox_reward_weight,
                'answer_reward_weight': self.answer_reward_weight,
                'predicted_bboxes': self.extract_bounding_boxes(predicted_reasoning or ""),
                'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                'invalid_format': False
            }
        }
        
        return result


def get_reward_model(config: Optional[Dict[str, Any]] = None) -> CorlL2SpatialReward:
    """
    Factory function to get the reward model instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized reward model
    """
    if config is None:
        config = {}
    
    return CorlL2SpatialReward(
        partial_credit=config.get('partial_credit', False),
        bbox_reward_weight=config.get('bbox_reward_weight', 0.3)
    )


# Convenience function for direct scoring
def compute_reward(model_output: str, ground_truth: str, partial_credit: bool = False, 
                   bbox_reward_weight: float = 0.3) -> float:
    """
    Convenience function to compute reward directly.
    
    Args:
        model_output: The model's output
        ground_truth: The ground truth
        partial_credit: Whether to enable partial credit
        bbox_reward_weight: Weight for bounding box IoU reward (0.0-1.0)
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    reward_model = CorlL2SpatialReward(
        partial_credit=partial_credit, 
        bbox_reward_weight=bbox_reward_weight
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
    print("Testing CORL Level 2 Spatial Understanding Reward Function")
    print("=" * 60)
    
    # Initialize reward model
    reward_model = CorlL2SpatialReward(partial_credit=True, bbox_reward_weight=0.3)
    
    # Test case 1: Correct answer with reasoning
    test1_output = """
    <think>
    Looking at both robot perspectives to find which object is farthest from the cup.
    I can see a cup located at [100, 150, 200, 250] in the main perspective.
    I can see a tomato positioned at coordinates [300, 100, 400, 200].
    I notice a bowl at [500, 300, 600, 400].
    
    Distance from cup to tomato: 223.6 pixels
    Distance from cup to bowl: 458.3 pixels
    
    Based on the spatial analysis, bowl is located farthest from cup.
    </think>
    
    \\boxed{bowl}
    """
    test1_gt = """
    <think>
    I can see a cup at bounding box [100, 150, 200, 250].
    I can see a tomato at [300, 100, 400, 200].
    I can see a bowl at [500, 300, 600, 400].
    
    Calculating distances from cup:
    - Distance to tomato: 223.6 pixels
    - Distance to bowl: 458.3 pixels
    
    Therefore, bowl is farthest from cup.
    </think>
    
    \\boxed{bowl}
    """
    
    result1 = reward_model.reward(test1_output, test1_gt)
    print(f"Test 1 - Perfect match:")
    print(f"  Score: {result1['score']:.3f}")
    print(f"  Predicted: {result1['predicted_answer']}")
    print(f"  Ground Truth: {result1['ground_truth_answer']}")
    print(f"  Answer Score: {result1['answer_score']:.3f}")
    print(f"  Bbox Score: {result1['bbox_score']:.3f}")
    print()
    
    # Test case 2: Correct answer without reasoning
    test2_output = "\\boxed{tomato}"
    test2_gt = "\\boxed{tomato}"
    
    result2 = reward_model.reward(test2_output, test2_gt)
    print(f"Test 2 - Correct answer only:")
    print(f"  Score: {result2['score']:.3f}")
    print(f"  Has Reasoning: {result2['has_reasoning']}")
    print()
    
    # Test case 3: Wrong answer
    test3_output = "\\boxed{cup}"
    test3_gt = "\\boxed{bowl}"
    
    result3 = reward_model.reward(test3_output, test3_gt)
    print(f"Test 3 - Wrong answer:")
    print(f"  Score: {result3['score']:.3f}")
    print(f"  Predicted: {result3['predicted_answer']}")
    print(f"  Ground Truth: {result3['ground_truth_answer']}")
    print()
    
    # Test case 4: Partial credit (similar objects)
    test4_output = "\\boxed{mug}"
    test4_gt = "\\boxed{cup}"
    
    result4 = reward_model.reward(test4_output, test4_gt)
    print(f"Test 4 - Similar objects (partial credit):")
    print(f"  Score: {result4['score']:.3f}")
    print(f"  Normalized Pred: {result4['details']['normalized_prediction']}")
    print(f"  Normalized GT: {result4['details']['normalized_ground_truth']}")
    print()
    
    # Test case 5: Invalid format
    test5_output = "The farthest object from the cup is the bowl on the table"
    test5_gt = "\\boxed{bowl}"
    
    result5 = reward_model.reward(test5_output, test5_gt)
    print(f"Test 5 - Invalid format:")
    print(f"  Score: {result5['score']:.3f}")
    print(f"  Invalid Format: {result5['details']['invalid_format']}")
    print()
    
    # Test case 6: Object name normalization
    test6_output = "\\boxed{tomato_1}"
    test6_gt = "\\boxed{tomato}"
    
    result6 = reward_model.reward(test6_output, test6_gt)
    print(f"Test 6 - Object name normalization:")
    print(f"  Score: {result6['score']:.3f}")
    print(f"  Normalized Pred: {result6['details']['normalized_prediction']}")
    print(f"  Normalized GT: {result6['details']['normalized_ground_truth']}")
    print()
    
    # Test bounding box extraction
    print("=" * 60)
    print("Testing Bounding Box Extraction:")
    print("=" * 60)
    
    test_text = """
    I can see a cup located at [100, 200, 300, 400].
    There's a bowl positioned at coordinates [500, 600, 700, 800].
    Distance from cup to bowl: 412.3 pixels
    I notice a tomato at [200, 100, 350, 250].
    """
    
    extracted = reward_model.extract_bounding_boxes(test_text)
    print(f"Extracted bboxes: {extracted}")
    
    # Test bbox reward calculation
    pred_reasoning = "I can see a cup at [100, 100, 200, 200]. I notice a bowl at [300, 300, 400, 400]."
    gt_reasoning = "I can see a cup located at [105, 105, 205, 205]. There's a bowl positioned at [295, 295, 395, 395]."
    
    bbox_score = reward_model.compute_bbox_reward(pred_reasoning, gt_reasoning)
    print(f"Bbox reward for similar boxes: {bbox_score:.3f}")
    print()
    
    # Test different reward weights
    print("=" * 60)
    print("Testing Different Reward Weights:")
    print("=" * 60)
    
    for weight in [0.0, 0.3, 0.5, 1.0]:
        test_model = CorlL2SpatialReward(partial_credit=True, bbox_reward_weight=weight)
        result = test_model.reward(test1_output, test1_gt)
        print(f"Bbox weight {weight:.1f}: Score = {result['score']:.3f} (Answer: {result['answer_score']:.3f}, Bbox: {result['bbox_score']:.3f})")