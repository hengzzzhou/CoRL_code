"""
Reward function for CORL real dual-robot perspective spatial reasoning tasks
"""

import re
from typing import Dict, Any, Optional, List, Tuple


class CorlRealSpatialReward:
    """
    Reward model for evaluating real spatial reasoning responses from dual robot perspectives.
    
    The model expects responses in the following format:
    <think>
    [reasoning process]
    </think>
    
    \boxed{answer}
    
    This handles real-world spatial reasoning tasks with various question types and answer formats.
    """
    
    def __init__(self, partial_credit: bool = False, bbox_reward_weight: float = 0.2):
        """
        Initialize the reward model.
        
        Args:
            partial_credit: Whether to give partial credit for partially correct answers
            bbox_reward_weight: Weight for reasoning quality reward (0.0-1.0)
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
                
                # Validate that the answer is reasonable
                if self._is_valid_answer_format(extracted_answer):
                    result['answer'] = extracted_answer.lower()
                    break
        
        # Fallback: try to find answer after "Answer:" or similar keywords
        if result['answer'] is None:
            answer_patterns = [
                r'(?:Answer|Final answer|The answer is)[:\s]+([^\n.!?]+)',
                r'(?:Therefore|So|Thus)[,:\s]+([^\n.!?]+)',
                r'(?:I think|I believe)[:\s]+([^\n.!?]+)'
            ]
            
            for pattern in answer_patterns:
                answer_match = re.search(pattern, output_str, re.IGNORECASE)
                if answer_match:
                    candidate = answer_match.group(1).strip()
                    if self._is_valid_answer_format(candidate):
                        result['answer'] = candidate.lower()
                        break
        
        return result
    
    def _is_valid_answer_format(self, answer: str) -> bool:
        """
        Validate if the extracted answer is reasonable.
        
        Args:
            answer: The extracted answer string
            
        Returns:
            True if the answer format is valid, False otherwise
        """
        if not answer or len(answer.strip()) == 0:
            return False
        
        answer = answer.strip().lower()
        
        # Check length constraints
        if len(answer) > 100:  # Too long to be a reasonable answer
            return False
        
        # Should not contain excessive newlines or special formatting
        if answer.count('\n') > 2:
            return False
        
        # Should not be pure numbers/coordinates unless it's a counting task
        number_only_pattern = r'^\s*[\d\s,.-]+\s*$'
        if re.match(number_only_pattern, answer) and len(answer) > 10:
            return False  # Likely coordinate data, not an answer
        
        # Should not start with common question words (indicates incomplete parsing)
        invalid_starts = ['what', 'which', 'how', 'where', 'when', 'why']
        first_word = answer.split()[0] if answer.split() else ''
        if first_word in invalid_starts:
            return False
        
        return True
    
    def extract_object_mentions(self, text: str) -> List[str]:
        """
        Extract object mentions from reasoning text for analysis.
        
        Args:
            text: The reasoning text
            
        Returns:
            List of object names mentioned
        """
        if not text:
            return []
        
        objects = []
        
        # Patterns to find object mentions
        patterns = [
            r'(?:I can see|I notice|I spot|There is|There\'s)\s+(?:a|an|the)?\s*(\w+)',
            r'(?:the|a|an)\s+(\w+)\s+(?:is|appears|looks|seems)',
            r'(\w+)\s+(?:object|item|thing)',
            r'detect(?:ed)?\s+(?:a|an|the)?\s*(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                obj_name = match.group(1).lower()
                # Filter out common non-object words
                if obj_name not in ['image', 'robot', 'perspective', 'view', 'scene', 'environment']:
                    objects.append(obj_name)
        
        return list(set(objects))  # Remove duplicates
    
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
        patterns = [
            # "I can see a cup located at [x1, y1, x2, y2]"
            r'(?:I\s+(?:can\s+see|notice|spot)\s+a\s+|There\'?s\s+a\s+)(\w+)\s+(?:located\s+|positioned\s+)?at\s+(?:coordinates\s+|bounding\s+box\s+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            # "cup is visible at [x1, y1, x2, y2]"
            r'(\w+)\s+is\s+visible\s+at\s+\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
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
    
    def extract_spatial_relationships(self, text: str) -> List[str]:
        """
        Extract spatial relationship mentions from reasoning text.
        
        Args:
            text: The reasoning text
            
        Returns:
            List of spatial relationships mentioned
        """
        if not text:
            return []
        
        relationships = []
        
        # Patterns for spatial relationships
        spatial_patterns = [
            r'(\w+)\s+(?:is|appears)\s+(?:to the\s+)?(?:left|right|above|below|near|far|close|distant)',
            r'(?:left|right|above|below|near|far|close|distant)\s+(?:of|from)\s+(?:the\s+)?(\w+)',
            r'(\w+)\s+(?:and\s+)?(\w+)\s+are\s+(?:close|near|far|distant)',
            r'distance\s+(?:between|from)\s+(\w+)\s+(?:to|and)\s+(\w+)',
        ]
        
        for pattern in spatial_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationship = match.group(0).strip()
                if len(relationship) < 100:  # Reasonable length
                    relationships.append(relationship.lower())
        
        return relationships
    
    def compute_reasoning_quality(self, predicted_reasoning: str, ground_truth_reasoning: str) -> float:
        """
        Compute reward based on reasoning quality.
        
        Args:
            predicted_reasoning: The model's reasoning text
            ground_truth_reasoning: The ground truth reasoning text
            
        Returns:
            Reasoning quality score between 0.0 and 1.0
        """
        if not predicted_reasoning or not ground_truth_reasoning:
            return 0.0
        
        # Extract objects and relationships from both texts
        pred_objects = set(self.extract_object_mentions(predicted_reasoning))
        gt_objects = set(self.extract_object_mentions(ground_truth_reasoning))
        
        pred_relationships = set(self.extract_spatial_relationships(predicted_reasoning))
        gt_relationships = set(self.extract_spatial_relationships(ground_truth_reasoning))
        
        # Calculate object mention overlap
        if gt_objects:
            object_overlap = len(pred_objects & gt_objects) / len(gt_objects)
        else:
            object_overlap = 1.0 if not pred_objects else 0.5
        
        # Calculate spatial relationship overlap
        if gt_relationships:
            relationship_overlap = len(pred_relationships & gt_relationships) / len(gt_relationships)
        else:
            relationship_overlap = 1.0 if not pred_relationships else 0.5
        
        # Check for key reasoning terms
        reasoning_terms = [
            'perspective', 'viewpoint', 'analyze', 'observe', 'spatial', 'relationship',
            'position', 'location', 'distance', 'closer', 'farther', 'left', 'right',
            'above', 'below', 'near', 'far'
        ]
        
        pred_terms = sum(1 for term in reasoning_terms if term in predicted_reasoning.lower())
        gt_terms = sum(1 for term in reasoning_terms if term in ground_truth_reasoning.lower())
        
        if gt_terms > 0:
            reasoning_term_score = min(pred_terms / gt_terms, 1.0)
        else:
            reasoning_term_score = 0.5
        
        # Weighted combination
        quality_score = (
            0.4 * object_overlap +
            0.4 * relationship_overlap +
            0.2 * reasoning_term_score
        )
        
        return min(quality_score, 1.0)
    
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
        
        # Convert to lowercase and strip whitespace
        answer = answer.lower().strip()
        
        # Remove common articles and prefixes
        answer = re.sub(r'^(a\s+|an\s+|the\s+)', '', answer)
        
        # Remove common punctuation
        answer = re.sub(r'[.,;!?]', '', answer)
        
        # Handle yes/no variations
        yes_variations = ['yes', 'true', 'correct', 'affirmative', 'positive', 'right']
        no_variations = ['no', 'false', 'incorrect', 'negative', 'wrong']
        
        if answer in yes_variations:
            return 'yes'
        elif answer in no_variations:
            return 'no'
        
        # Normalize numbers
        answer = re.sub(r'\b(\d+)\b', lambda m: str(int(m.group(1))), answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
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
        pred_norm = self.normalize_answer(predicted_answer)
        gt_norm = self.normalize_answer(ground_truth)
        
        # Check for exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # If partial credit is enabled, check for partial matches
        if self.partial_credit:
            # Check if answers contain each other
            if gt_norm and pred_norm:
                if gt_norm in pred_norm or pred_norm in gt_norm:
                    return 0.7
            
            # Check for numeric answers (counting tasks)
            pred_numbers = re.findall(r'\d+', pred_norm)
            gt_numbers = re.findall(r'\d+', gt_norm)
            
            if pred_numbers and gt_numbers:
                # For counting/numeric tasks, partial credit based on proximity
                try:
                    pred_num = int(pred_numbers[0])
                    gt_num = int(gt_numbers[0])
                    
                    if abs(pred_num - gt_num) == 0:
                        return 1.0
                    elif abs(pred_num - gt_num) == 1:
                        return 0.8
                    elif abs(pred_num - gt_num) <= 2:
                        return 0.6
                    elif abs(pred_num - gt_num) <= 3:
                        return 0.4
                except ValueError:
                    pass
            
            # Check for yes/no questions
            if (pred_norm in ['yes', 'no'] and gt_norm in ['yes', 'no']):
                return 0.0  # Wrong yes/no is completely wrong
            
            # Check for similar object names
            if len(pred_norm.split()) == 1 and len(gt_norm.split()) == 1:
                # Single word answers - check for similarity
                pred_word = pred_norm.split()[0] if pred_norm.split() else ''
                gt_word = gt_norm.split()[0] if gt_norm.split() else ''
                
                if pred_word and gt_word:
                    # Simple string similarity
                    if pred_word.startswith(gt_word[:3]) or gt_word.startswith(pred_word[:3]):
                        return 0.5
        
        return 0.0
    
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
                - 'reasoning_score': The reasoning quality score
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
                'reasoning_score': 0.0,
                'answer_score': 0.0,
                'details': {
                    'normalized_prediction': None,
                    'normalized_ground_truth': self.normalize_answer(gt_answer),
                    'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                    'reasoning_reward_weight': self.bbox_reward_weight,
                    'answer_reward_weight': self.answer_reward_weight,
                    'predicted_objects': [],
                    'ground_truth_objects': self.extract_object_mentions(gt_reasoning or ""),
                    'predicted_relationships': [],
                    'ground_truth_relationships': self.extract_spatial_relationships(gt_reasoning or ""),
                    'predicted_bboxes': self.extract_bounding_boxes(predicted_reasoning or ""),
                    'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                    'invalid_format': True
                }
            }
            return result
        
        # Compute answer reward score (main component)
        answer_score = self.compute_answer_reward(predicted_answer, gt_answer)
        
        # Compute reasoning quality score (secondary component)
        reasoning_score = 0.0
        bbox_score = 0.0
        if self.bbox_reward_weight > 0 and predicted_reasoning and gt_reasoning:
            reasoning_score = self.compute_reasoning_quality(predicted_reasoning, gt_reasoning)
            bbox_score = self.compute_bbox_reward(predicted_reasoning, gt_reasoning)
            # Combine reasoning quality and bbox IoU
            reasoning_score = (reasoning_score + bbox_score) / 2.0
        
        # Compute weighted final score
        if self.bbox_reward_weight > 0:
            final_score = (self.answer_reward_weight * answer_score + 
                          self.bbox_reward_weight * reasoning_score)
        else:
            final_score = answer_score
        
        # Prepare result
        result = {
            'score': final_score,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': gt_answer,
            'has_reasoning': predicted_reasoning is not None,
            'reasoning_score': reasoning_score,
            'answer_score': answer_score,
            'details': {
                'normalized_prediction': self.normalize_answer(predicted_answer),
                'normalized_ground_truth': self.normalize_answer(gt_answer),
                'reasoning_length': len(predicted_reasoning) if predicted_reasoning else 0,
                'reasoning_reward_weight': self.bbox_reward_weight,
                'answer_reward_weight': self.answer_reward_weight,
                'predicted_objects': self.extract_object_mentions(predicted_reasoning or ""),
                'ground_truth_objects': self.extract_object_mentions(gt_reasoning or ""),
                'predicted_relationships': self.extract_spatial_relationships(predicted_reasoning or ""),
                'ground_truth_relationships': self.extract_spatial_relationships(gt_reasoning or ""),
                'predicted_bboxes': self.extract_bounding_boxes(predicted_reasoning or ""),
                'ground_truth_bboxes': self.extract_bounding_boxes(gt_reasoning or ""),
                'invalid_format': False
            }
        }
        
        return result


def get_reward_model(config: Optional[Dict[str, Any]] = None) -> CorlRealSpatialReward:
    """
    Factory function to get the reward model instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized reward model
    """
    if config is None:
        config = {}
    
    return CorlRealSpatialReward(
        partial_credit=config.get('partial_credit', False),
        bbox_reward_weight=config.get('bbox_reward_weight', 0.2)
    )


# Convenience function for direct scoring
def compute_reward(model_output: str, ground_truth: str, partial_credit: bool = False, 
                   bbox_reward_weight: float = 0.2) -> float:
    """
    Convenience function to compute reward directly.
    
    Args:
        model_output: The model's output
        ground_truth: The ground truth
        partial_credit: Whether to enable partial credit
        bbox_reward_weight: Weight for reasoning quality reward (0.0-1.0)
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    reward_model = CorlRealSpatialReward(
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
    return compute_reward(solution_str, ground_truth, partial_credit=True, bbox_reward_weight=0.2)


if __name__ == "__main__":
    # Test cases
    print("Testing CORL Real Spatial Reasoning Reward Function")
    print("=" * 60)
    
    # Initialize reward model
    reward_model = CorlRealSpatialReward(partial_credit=True, bbox_reward_weight=0.2)
    
    # Test case 1: Perfect match with reasoning
    test1_output = """
    <think>
    Looking at both robot perspectives to answer the question.
    I can observe a cup and a bowl in the main perspective.
    I notice the cup is positioned to the left of the bowl.
    From the auxiliary perspective, I can see the same spatial relationship.
    The distance between cup and bowl appears consistent across both views.
    </think>
    
    \\boxed{cup}
    """
    test1_gt = """
    <think>
    Analyzing the spatial relationships in both images.
    I can see a cup and bowl in the scene.
    The cup is positioned to the left of the bowl.
    This relationship is consistent across both robot perspectives.
    </think>
    
    \\boxed{cup}
    """
    
    result1 = reward_model.reward(test1_output, test1_gt)
    print(f"Test 1 - Perfect match:")
    print(f"  Score: {result1['score']:.3f}")
    print(f"  Answer Score: {result1['answer_score']:.3f}")
    print(f"  Reasoning Score: {result1['reasoning_score']:.3f}")
    print(f"  Objects Found: {result1['details']['predicted_objects']}")
    print()
    
    # Test case 2: Correct answer, different reasoning
    test2_output = "\\boxed{yes}"
    test2_gt = "\\boxed{yes}"
    
    result2 = reward_model.reward(test2_output, test2_gt)
    print(f"Test 2 - Correct answer only:")
    print(f"  Score: {result2['score']:.3f}")
    print(f"  Has Reasoning: {result2['has_reasoning']}")
    print()
    
    # Test case 3: Wrong answer
    test3_output = "\\boxed{no}"
    test3_gt = "\\boxed{yes}"
    
    result3 = reward_model.reward(test3_output, test3_gt)
    print(f"Test 3 - Wrong answer:")
    print(f"  Score: {result3['score']:.3f}")
    print()
    
    # Test case 4: Counting task with close number
    test4_output = "\\boxed{3}"
    test4_gt = "\\boxed{4}"
    
    result4 = reward_model.reward(test4_output, test4_gt)
    print(f"Test 4 - Close counting:")
    print(f"  Score: {result4['score']:.3f}")
    print(f"  Normalized Pred: {result4['details']['normalized_prediction']}")
    print(f"  Normalized GT: {result4['details']['normalized_ground_truth']}")
    print()
    
    # Test case 5: Invalid format
    test5_output = "I think the answer is probably the cup based on the images"
    test5_gt = "\\boxed{cup}"
    
    result5 = reward_model.reward(test5_output, test5_gt)
    print(f"Test 5 - Invalid format:")
    print(f"  Score: {result5['score']:.3f}")
    print(f"  Invalid Format: {result5['details']['invalid_format']}")
    print()
    
    # Test case 6: Partial match
    test6_output = "\\boxed{the cup}"
    test6_gt = "\\boxed{cup}"
    
    result6 = reward_model.reward(test6_output, test6_gt)
    print(f"Test 6 - Partial match (article):")
    print(f"  Score: {result6['score']:.3f}")
    print(f"  Normalized Pred: '{result6['details']['normalized_prediction']}'")
    print(f"  Normalized GT: '{result6['details']['normalized_ground_truth']}'")
    print()
    
    # Test object and relationship extraction
    print("=" * 60)
    print("Testing Object and Relationship Extraction:")
    print("=" * 60)
    
    test_text = """
    I can see a cup and a bowl in the main perspective.
    The cup is positioned to the left of the bowl.
    I notice there is also a tomato near the bowl.
    Distance from cup to bowl appears to be about 20cm.
    """
    
    objects = reward_model.extract_object_mentions(test_text)
    relationships = reward_model.extract_spatial_relationships(test_text)
    
    print(f"Objects: {objects}")
    print(f"Relationships: {relationships}")
    print()
    
    # Test different reasoning weights
    print("=" * 60)
    print("Testing Different Reasoning Weights:")
    print("=" * 60)
    
    for weight in [0.0, 0.2, 0.5, 0.8]:
        test_model = CorlRealSpatialReward(partial_credit=True, bbox_reward_weight=weight)
        result = test_model.reward(test1_output, test1_gt)
        print(f"Reasoning weight {weight:.1f}: Score = {result['score']:.3f} (Answer: {result['answer_score']:.3f}, Reasoning: {result['reasoning_score']:.3f})")