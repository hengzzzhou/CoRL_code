"""
Reward function for dual-robot perspective spatial reasoning tasks
"""

import re
from typing import Union, Dict, Any, Optional, List, Tuple
import json


class DualRobotReasoningReward:
    """
    Reward model for evaluating spatial reasoning responses from dual robot perspectives.
    
    The model expects responses in the following format:
    <think>
    [reasoning process]
    </think>
    
    \boxed{answer}
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
        # Handle both \boxed{} and \\boxed{}
        boxed_patterns = [
            r'\\\\boxed\{([^}]*)\}',  # \\boxed{answer}
            r'\\boxed\{([^}]*)\}',     # \boxed{answer}
            r'\$\\boxed\{([^}]*)\}\$', # $\boxed{answer}$
        ]
        
        for pattern in boxed_patterns:
            boxed_match = re.search(pattern, output_str)
            if boxed_match:
                extracted_answer = boxed_match.group(1).strip()
                
                # Validate that the answer is a reasonable format
                # Should be a number, not a list of coordinates
                if self._is_valid_answer_format(extracted_answer):
                    result['answer'] = extracted_answer.lower()
                    break
        
        # Fallback: try to find answer after "Answer:" or similar keywords
        if result['answer'] is None:
            answer_pattern = r'(?:Answer|Final answer|The answer is)[:\s]+([^\n.]+)'
            answer_match = re.search(answer_pattern, output_str, re.IGNORECASE)
            if answer_match:
                result['answer'] = answer_match.group(1).strip().lower()
        
        return result
    
    def _is_valid_answer_format(self, answer: str) -> bool:
        """
        Validate if the extracted answer is in a reasonable format.
        
        Args:
            answer: The extracted answer string
            
        Returns:
            True if the answer format is valid, False otherwise
        """
        if not answer or len(answer.strip()) == 0:
            return False
        
        answer = answer.strip()
        
        # Check for obvious invalid formats
        # 1. Contains too many bracket coordinates (likely bbox coords)
        bracket_count = answer.count('[') + answer.count(']')
        if bracket_count > 4:  # More than 2 coordinate pairs suggests invalid format
            return False
        
        # 2. Contains multiple number sequences separated by spaces (likely coordinates)
        import re
        number_sequences = re.findall(r'\d+', answer)
        if len(number_sequences) > 3:  # More than 3 numbers suggests coordinate list
            return False
        
        # 3. Check for reasonable answer patterns
        # Should be a simple number, word, or short phrase
        if len(answer) > 20:  # Too long to be a reasonable answer
            return False
        
        # 4. Should not contain excessive whitespace or newlines
        if answer.count('\n') > 1 or answer.count('  ') > 2:
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
        
        # Single comprehensive pattern to match all bbox coordinate formats
        # This pattern matches: object_name + various connectors + [x1, y1, x2, y2]
        pattern = r'(?:I\s+(?:can\s+see|notice|spot)\s+a\s+|There\'?s\s+a\s+|Additionally,?\s+there\'?s\s+(?:a\s+|another\s+)?|I\s+also\s+notice\s+a\s+(?:second\s+|third\s+)?)?(\w+)\s+(?:located\s+|positioned\s+|is\s+visible\s+)?at\s+(?:coordinates\s+|bounding\s+box\s+|position\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            object_name = groups[0].lower()
            coords = [int(groups[i]) for i in range(1, 5)]
            bboxes.append((object_name, coords))
        
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
        
        # Group bounding boxes by object type
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
        
        # Calculate IoU for each object type using optimal matching
        total_ious = []
        all_object_types = set(pred_by_type.keys()) | set(gt_by_type.keys())
        
        for obj_type in all_object_types:
            pred_boxes = pred_by_type.get(obj_type, [])
            gt_boxes = gt_by_type.get(obj_type, [])
            
            if not pred_boxes or not gt_boxes:
                total_ious.append(0.0)  # Missing predictions or ground truth
                continue
            
            # Use optimal assignment for multiple boxes of same type
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
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison.
        
        Args:
            answer: The answer string to normalize
            
        Returns:
            Normalized answer string
        """
        if answer is None:
            return ""
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove common punctuation
        answer = re.sub(r'[.,;!?]', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Handle yes/no variations
        yes_variations = ['yes', 'true', 'correct', 'affirmative', 'y']
        no_variations = ['no', 'false', 'incorrect', 'negative', 'n']
        
        if answer in yes_variations:
            return 'yes'
        elif answer in no_variations:
            return 'no'
        
        return answer
    
    def compute_reward(self, predicted_answer: str, ground_truth: str) -> float:
        """
        Compute the reward score for a predicted answer.
        
        Args:
            predicted_answer: The model's predicted answer
            ground_truth: The correct answer
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        # Normalize both answers
        pred_norm = self.normalize_answer(predicted_answer)
        gt_norm = self.normalize_answer(ground_truth)
        
        # Check for exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # If partial credit is enabled, check for partial matches
        if self.partial_credit:
            # Check if the ground truth is contained in the prediction
            if gt_norm and gt_norm in pred_norm:
                return 0.5
            
            # Check for semantic similarity in yes/no questions
            yes_words = ['yes', 'true', 'correct']
            no_words = ['no', 'false', 'incorrect']
            
            if (any(word in pred_norm for word in yes_words) and 
                any(word in gt_norm for word in yes_words)):
                return 0.8
            
            if (any(word in pred_norm for word in no_words) and 
                any(word in gt_norm for word in no_words)):
                return 0.8
        
        return 0.0
    
    def reward(self, model_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Main reward function that evaluates model output against ground truth.
        
        Args:
            model_output: The complete model output string
            ground_truth: The ground truth string (may include reasoning and answer)
            
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
        
        # Parse ground truth (it might also be in the same format)
        parsed_gt = self.parse_model_output(ground_truth)
        gt_answer = parsed_gt['answer']
        gt_reasoning = parsed_gt['reasoning']
        
        # If ground truth doesn't have structured format, treat it as plain answer
        if gt_answer is None:
            gt_answer = ground_truth.strip()
        
        # Check if answer format is valid first
        if predicted_answer is None or not self._is_valid_answer_format(predicted_answer):
            # If answer format is invalid, return 0 score immediately
            result = {
                'score': 0.0,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': gt_answer,
                'has_reasoning': predicted_reasoning is not None,
                'bbox_score': 0.0,
                'answer_score': 0.0,
                'details': {
                    'normalized_prediction': None,
                    'normalized_ground_truth': self.normalize_answer(gt_answer),
                    'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                    'bbox_reward_weight': self.bbox_reward_weight,
                    'answer_reward_weight': self.answer_reward_weight,
                    'predicted_bboxes': [],
                    'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                    'invalid_format': True
                }
            }
            return result
        
        # Compute answer reward score
        answer_score = self.compute_reward(predicted_answer, gt_answer)
        
        # Compute bounding box reward score
        bbox_score = 0.0
        if self.bbox_reward_weight > 0 and predicted_reasoning and gt_reasoning:
            bbox_score = self.compute_bbox_reward(predicted_reasoning, gt_reasoning)
        
        # Compute weighted final score
        if self.bbox_reward_weight > 0:
            final_score = (self.answer_reward_weight * answer_score + 
                          self.bbox_reward_weight * bbox_score)
        else:
            final_score = answer_score
        if gt_answer==0 and final_score==1:
            final_score=0.7
        # Prepare result
        result = {
            'score': final_score,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': gt_answer,
            'has_reasoning': predicted_reasoning is not None,
            'bbox_score': bbox_score,
            'answer_score': answer_score,
            'details': {
                'normalized_prediction': self.normalize_answer(predicted_answer),
                'normalized_ground_truth': self.normalize_answer(gt_answer),
                'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                'bbox_reward_weight': self.bbox_reward_weight,
                'answer_reward_weight': self.answer_reward_weight,
                'predicted_bboxes': self.extract_bounding_boxes(predicted_reasoning or ""),
                'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                'invalid_format': False
            }
        }
        
        return result


def get_reward_model(config: Optional[Dict[str, Any]] = None) -> DualRobotReasoningReward:
    """
    Factory function to get the reward model instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized reward model
    """
    if config is None:
        config = {}
    
    return DualRobotReasoningReward(
        partial_credit=config.get('partial_credit', False),
        bbox_reward_weight=config.get('bbox_reward_weight', 0.3)
    )


# Convenience function for direct scoring
def compute_reward(model_output: str, ground_truth: str, partial_credit: bool = False, bbox_reward_weight: float = 0.3) -> float:
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
    reward_model = DualRobotReasoningReward(partial_credit=partial_credit, bbox_reward_weight=bbox_reward_weight)
    result = reward_model.reward(model_output, ground_truth)
    # print(result)
    # print(result['score'])
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
    print("Testing Dual Robot Reasoning Reward Function")
    print("=" * 50)
    
    # Initialize reward model with bbox reward
    reward_model = DualRobotReasoningReward(partial_credit=True, bbox_reward_weight=0.3)
    
    # Test case 1: Correct answer with reasoning
    test1_output = """
    <think>
    Looking at both robot perspectives, I can see that the object A is to the left 
    of object B in both views, confirming the spatial relationship.
    </think>
    
    \\boxed{yes}
    """
    test1_gt = "\\boxed{yes}"
    
    result1 = reward_model.reward(test1_output, test1_gt)
    print(f"Test 1 - Correct with reasoning:")
    print(f"  Score: {result1['score']}")
    print(f"  Predicted: {result1['predicted_answer']}")
    print(f"  Ground Truth: {result1['ground_truth_answer']}")
    print(f"  Has Reasoning: {result1['has_reasoning']}")
    print()
    
    # Test case 2: Wrong answer
    test2_output = "\\boxed{no}"
    test2_gt = "\\boxed{yes}"
    
    result2 = reward_model.reward(test2_output, test2_gt)
    print(f"Test 2 - Wrong answer:")
    print(f"  Score: {result2['score']}")
    print(f"  Predicted: {result2['predicted_answer']}")
    print(f"  Ground Truth: {result2['ground_truth_answer']}")
    print()
    
    # Test case 3: Missing format
    test3_output = "The answer is yes"
    test3_gt = "yes"
    
    result3 = reward_model.reward(test3_output, test3_gt)
    print(f"Test 3 - Alternative format:")
    print(f"  Score: {result3['score']}")
    print(f"  Predicted: {result3['predicted_answer']}")
    print(f"  Ground Truth: {result3['ground_truth_answer']}")
    print()
    
    # Test case 4: Complex spatial answer
    test4_output = """
    <think>
    Analyzing the two robot views, the sink appears at coordinates (398,70,804,326) 
    and the microwave at (920,0,1439,415). The sink is clearly to the left.
    </think>
    
    \\boxed{yes}
    """
    test4_gt = """
    <think>
    I can identify the sink at bounding box (398,70,804,326) and the microwave at 
    bounding box (920,0,1439,415). Based on their positions, the sink is located 
    to the left of the microwave.
    </think>
    
    \\boxed{yes}
    """
    
    result4 = reward_model.reward(test4_output, test4_gt)
    print(f"Test 4 - Both with reasoning:")
    print(f"  Score: {result4['score']}")
    print(f"  Predicted: {result4['predicted_answer']}")
    print(f"  Ground Truth: {result4['ground_truth_answer']}")
    print(f"  Has Reasoning: {result4['has_reasoning']}")
    print(f"  Answer Score: {result4['answer_score']:.3f}")
    print(f"  Bbox Score: {result4['bbox_score']:.3f}")
    print()
    
    # Test bounding box extraction and IoU
    print("=" * 50)
    print("Testing Bounding Box Features:")
    print("=" * 50)
    
    test_text = "I can see a cup at [100, 200, 300, 400] and a tomato located at [500, 600, 700, 800]."
    extracted = reward_model.extract_bounding_boxes(test_text)
    print(f"Extracted bboxes: {extracted}")
    
    # Test IoU calculation
    box1, box2 = [100, 100, 200, 200], [150, 150, 250, 250]
    iou = reward_model.calculate_iou(box1, box2)
    print(f"IoU between {box1} and {box2}: {iou:.3f}")
    
    # Test reward with bbox
    reward_score = reward_model.compute_bbox_reward(
        "I see a cup at [100, 200, 300, 400]",
        "I see a cup at [105, 205, 305, 405]"
    )
    print(f"Bbox reward for similar boxes: {reward_score:.3f}")
    print()
    
    # Test optimal matching with multiple boxes
    print("=" * 60)
    print("Testing Optimal Matching with Multiple Boxes:")
    print("=" * 60)
    
    # Test case: 3 predicted vs 3 ground truth (different order) - exact format
    pred_reasoning = "I can see a tomato at [100, 100, 200, 200]. I can see a tomato at [300, 300, 400, 400]. I can see a tomato at [500, 500, 600, 600]."
    gt_reasoning = "I can see a tomato at [510, 510, 610, 610]. I can see a tomato at [110, 110, 210, 210]. I can see a tomato at [310, 310, 410, 410]."
    
    bbox_score = reward_model.compute_bbox_reward(pred_reasoning, gt_reasoning)
    print(f"Multi-box optimal matching score: {bbox_score:.3f}")
    
    # Show the extracted boxes
    pred_boxes = reward_model.extract_bounding_boxes(pred_reasoning)
    gt_boxes = reward_model.extract_bounding_boxes(gt_reasoning)
    print(f"Predicted boxes ({len(pred_boxes)}): {pred_boxes}")
    print(f"Ground truth boxes ({len(gt_boxes)}): {gt_boxes}")
    
    # Demonstrate optimal assignment vs greedy
    print("\\nManual IoU calculations:")
    for i, (pred_name, pred_box) in enumerate(pred_boxes):
        print(f"  Pred {i}: {pred_box}")
        for j, (gt_name, gt_box) in enumerate(gt_boxes):
            if pred_name == gt_name:  # Same object type
                iou = reward_model.calculate_iou(pred_box, gt_box)
                print(f"    vs GT {j} {gt_box}: IoU = {iou:.3f}")
    
    # Test optimal assignment directly
    if len(pred_boxes) == 3 and len(gt_boxes) == 3:
        pred_coords = [box[1] for box in pred_boxes]
        gt_coords = [box[1] for box in gt_boxes]
        assignment_score = reward_model.optimal_bbox_matching(pred_coords, gt_coords)
        print(f"\\nOptimal assignment score: {assignment_score:.3f}")
        
        # Show what the optimal assignment would be
        cost_matrix = []
        for pred_box in pred_coords:
            row = []
            for gt_box in gt_coords:
                iou = reward_model.calculate_iou(pred_box, gt_box)
                row.append(-iou)
            cost_matrix.append(row)
        
        assignments = reward_model.hungarian_algorithm(cost_matrix)
        print("Optimal assignments:")
        for pred_idx, gt_idx in assignments:
            pred_box = pred_coords[pred_idx]
            gt_box = gt_coords[gt_idx]
            iou = reward_model.calculate_iou(pred_box, gt_box)
            print(f"  Pred {pred_idx} {pred_box} -> GT {gt_idx} {gt_box}, IoU: {iou:.3f}")
    
    print()
    
    # Test with different numbers of boxes (penalty case)
    pred_reasoning2 = "I can see a cup at [100, 100, 200, 200]. I can see a cup at [300, 300, 400, 400]."
    gt_reasoning2 = "I can see a cup at [105, 105, 205, 205]."
    
    bbox_score2 = reward_model.compute_bbox_reward(pred_reasoning2, gt_reasoning2)
    print(f"Mismatched count penalty (2 pred, 1 GT): {bbox_score2:.3f}")
    
    pred_boxes2 = reward_model.extract_bounding_boxes(pred_reasoning2)
    gt_boxes2 = reward_model.extract_bounding_boxes(gt_reasoning2)
    print(f"Pred: {len(pred_boxes2)} boxes, GT: {len(gt_boxes2)} boxes")
    print()
