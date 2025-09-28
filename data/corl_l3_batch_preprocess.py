"""
CORL-L3 Dataset Batch Preprocessing Script - Object Grasping Tasks
Adapted from corl_l2_batch_preprocess.py for processing multiple Spatial_Understand datasets
Processes Spatial_Understand_0 through Spatial_Understand_5 datasets for VERL training
Focuses on object grasping tasks with position-based selection
"""

import os
import json
import datasets
from datasets import Dataset
import argparse
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def image_to_binary(image_path, compress=False, quality=85, max_size=512):
    """Convert image to binary format with compression options
    
    Returns:
        tuple: (binary_data, scale_ratio) where scale_ratio is the scaling factor applied
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None, 1.0

        if compress:
            # Load, resize and compress image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if image is too large
                width, height = img.size
                original_max_dim = max(width, height)
                
                if original_max_dim > max_size:
                    ratio = max_size / original_max_dim
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height),
                                     Image.Resampling.LANCZOS)
                else:
                    ratio = 1.0

                # Compress to JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                return buffer.getvalue(), ratio
        else:
            # Original method - read file directly
            with open(image_path, 'rb') as f:
                return f.read(), 1.0

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None, 1.0


def scale_bbox(bbox_dict, scale_ratio):
    """Scale bounding box coordinates according to image scaling ratio"""
    if scale_ratio == 1.0:
        return bbox_dict
    
    scaled_bbox = {}
    for obj_name, bbox in bbox_dict.items():
        if bbox is not None:
            # bbox<: [x1, y1, x2, y2]
            scaled_bbox[obj_name] = [
                int(bbox[0] * scale_ratio),  # x1
                int(bbox[1] * scale_ratio),  # y1
                int(bbox[2] * scale_ratio),  # x2
                int(bbox[3] * scale_ratio)   # y2
            ]
        else:
            scaled_bbox[obj_name] = None
    return scaled_bbox


def get_bbox_center(bbox):
    """Calculate center point of bounding box"""
    if bbox is None:
        return None
    # bbox format: [x1, y1, x2, y2]
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return [center_x, center_y]


def convert_question_to_grasping_instruction(original_question, target_object):
    """Convert spatial reasoning question to grasping instruction"""
    # Extract the position description from the original question
    if "second position from the left" in original_question.lower():
        position_desc = "in the second position from the left on the table"
    elif "first position from the left" in original_question.lower():
        position_desc = "in the first position from the left on the table"
    elif "third position from the left" in original_question.lower():
        position_desc = "in the third position from the left on the table"
    elif "leftmost" in original_question.lower() or "left-most" in original_question.lower():
        position_desc = "in the leftmost position on the table"
    elif "rightmost" in original_question.lower() or "right-most" in original_question.lower():
        position_desc = "in the rightmost position on the table"
    elif "middle" in original_question.lower() or "center" in original_question.lower():
        position_desc = "in the middle position on the table"
    else:
        # Fallback: extract everything after "Which object is" and before "?"
        if "which object is" in original_question.lower():
            start_idx = original_question.lower().find("which object is") + len("which object is")
            end_idx = original_question.find("?")
            if end_idx != -1:
                position_desc = original_question[start_idx:end_idx].strip()
            else:
                position_desc = "at the specified location"
        else:
            position_desc = "at the specified location"
    
    # Create grasping instruction
    grasping_instruction = f"Please grasp the {target_object} {position_desc}."
    return grasping_instruction


def create_bbox_context(bbox_dict, image_name):
    """Create context description based on bounding box information"""
    context_lines = []
    for obj_name, bbox in bbox_dict.items():
        if bbox is not None:
            # bbox<: [x1, y1, x2, y2]
            context_lines.append(
                f"{obj_name} detected in {image_name} at coordinates {bbox}")
        else:
            context_lines.append(f"{obj_name} not visible in {image_name}")
    return context_lines


def sort_objects_by_position(bbox_dict, sort_direction="left_to_right"):
    """Sort objects by their positions (left to right or top to bottom)"""
    visible_objects = {obj_name: bbox for obj_name, bbox in bbox_dict.items() if bbox is not None}
    
    if not visible_objects:
        return []
    
    if sort_direction == "left_to_right":
        # Sort by x-coordinate (left edge)
        sorted_objects = sorted(visible_objects.items(), key=lambda x: x[1][0])
    elif sort_direction == "right_to_left":
        # Sort by x-coordinate (right edge) in reverse
        sorted_objects = sorted(visible_objects.items(), key=lambda x: x[1][2], reverse=True)
    elif sort_direction == "top_to_bottom":
        # Sort by y-coordinate (top edge)
        sorted_objects = sorted(visible_objects.items(), key=lambda x: x[1][1])
    else:  # bottom_to_top
        # Sort by y-coordinate (bottom edge) in reverse
        sorted_objects = sorted(visible_objects.items(), key=lambda x: x[1][3], reverse=True)
    
    return sorted_objects


def process_data_generic(data, split, dataset_name, dataset_dir, args):
    """Generic data processing function that can be used for both individual and merged processing"""
    # Object grasping task system prompt
    instruction_following = (
        "You are a robot analyzing two images from different perspectives of the same environment to "
        "determine the precise location for grasping a specific object. These images represent two distinct "
        "robot viewpoints that may show overlapping or complementary parts of the scene. "
        "FIRST analyze the spatial relationships and objects visible from each robot's perspective, "
        "understanding how the different viewpoints relate to each other. Think through the reasoning process "
        "as an internal monologue, considering what each robot can see and how their observations combine "
        "to solve the grasping task. "
        "When identifying objects RELEVANT TO THE GRASPING TASK, you must provide their bounding box coordinates "
        "in the format: [x1, y1, x2, y2] where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. "
        "For example: 'I can see a cup at bounding box [100, 150, 200, 250]'. "
        "Analyze the positions of all relevant objects and determine which object should be grasped based on "
        "the spatial description in the instruction. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer should specify BOTH the image number (0 for main perspective, 1 for auxiliary perspective) "
        "AND the CENTER COORDINATES of the target object's bounding box in the format: image_number, [x, y] "
        "and put in \\boxed{}."
    )
    
    processed_data = []
    skipped_count = 0
    
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            logger.info(f"Processing {split} sample {idx}/{len(data)} from {dataset_name}")
        
        try:
            # Extract image paths
            image_a_name = item['image_A']
            image_b_name = item['image_B']
            
            # Build full image paths (use compressed jpg files)
            image_a_path = os.path.join(dataset_dir, "images_compressed", 
                                        image_a_name.replace('.png', '.jpg'))
            image_b_path = os.path.join(dataset_dir, "images_compressed", 
                                        image_b_name.replace('.png', '.jpg'))
            
            # Check if image files exist
            if not os.path.exists(image_a_path) or not os.path.exists(image_b_path):
                logger.warning(f"Image files not found, skipping sample {idx}: {image_a_path} or {image_b_path}")
                skipped_count += 1
                continue
            
            # Load images as binary data
            image_binaries = []
            scale_ratios = []
            for img_path in [image_a_path, image_b_path]:
                img_bytes, scale_ratio = image_to_binary(
                    img_path,
                    compress=args.compress_images,
                    quality=args.image_quality,
                    max_size=args.max_image_size
                )
                if img_bytes is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    break
                image_binaries.append(img_bytes)
                scale_ratios.append(scale_ratio)
            
            if len(image_binaries) != 2:
                logger.warning(f"Could not load both images, skipping sample {idx}")
                skipped_count += 1
                continue
            
            # Extract bbox info and ground truth
            bbox_a_original = item['bbox_A']
            bbox_b_original = item['bbox_B']
            gt_info = item['gt']
            
            # Scale bounding boxes according to compression
            original_size = 1920  # Original image was 1920x1080
            intermediate_size = 768  # Images in folder are 768px  
            base_scale = intermediate_size / original_size  # 0.4
            
            total_scale_a = base_scale * scale_ratios[0] 
            total_scale_b = base_scale * scale_ratios[1]
            
            bbox_a = scale_bbox(bbox_a_original, total_scale_a)  
            bbox_b = scale_bbox(bbox_b_original, total_scale_b)
            
            original_question = gt_info['description']
            target_object = gt_info['ground_truth']
            
            # Convert question to grasping instruction
            grasping_instruction = convert_question_to_grasping_instruction(original_question, target_object)
            
            # Find the target object's bounding box and calculate center
            target_bbox = None
            target_center = None
            target_perspective = None
            
            if bbox_a.get(target_object) is not None:
                target_bbox = bbox_a[target_object]
                target_center = get_bbox_center(target_bbox)
                target_perspective = "main"
            elif bbox_b.get(target_object) is not None:
                target_bbox = bbox_b[target_object]
                target_center = get_bbox_center(target_bbox)
                target_perspective = "auxiliary"
            
            if target_center is None:
                logger.warning(f"Target object {target_object} not found in either image, skipping sample {idx}")
                skipped_count += 1
                continue
            
            # Build user message (dual image format)
            user_message = f"main picture:<image>\\nauxiliary picture:<image>\\nInstruction: {grasping_instruction}"
            
            # Build detailed chain-of-thought reasoning for grasping task
            cot_reasoning = f"I need to analyze images from two robot perspectives to determine the precise grasping location for the instruction: '{grasping_instruction}'.\\n\\n"
            
            # Add detailed bbox analysis to CoT
            cot_reasoning += "Let me first examine the main perspective (Image A):\\n"
            visible_a = [(obj_name, bbox) for obj_name, bbox in bbox_a.items() if bbox is not None]
            
            sentence_patterns = [
                "I can see a {obj_type} located at {bbox}",
                "There's a {obj_type} positioned at coordinates {bbox}", 
                "I notice a {obj_type} at {bbox}",
                "A {obj_type} is visible at {bbox}",
                "I spot a {obj_type} at position {bbox}"
            ]
            
            for i, (obj_name, bbox) in enumerate(visible_a):
                pattern = sentence_patterns[i % len(sentence_patterns)]
                center = get_bbox_center(bbox)
                cot_reasoning += f"- {pattern.format(obj_type=obj_name, bbox=bbox)} with center at [{center[0]:.1f}, {center[1]:.1f}]\\n"
            
            # Sort objects by position for spatial analysis
            if visible_a:
                sorted_objects_a = sort_objects_by_position({name: bbox for name, bbox in visible_a}, "left_to_right")
                if len(sorted_objects_a) > 1:
                    cot_reasoning += "\\nSpatial arrangement from left to right in main perspective:\\n"
                    for pos, (obj_name, bbox) in enumerate(sorted_objects_a, 1):
                        center = get_bbox_center(bbox)
                        cot_reasoning += f"- Position {pos}: {obj_name} at center [{center[0]:.1f}, {center[1]:.1f}]\\n"
            
            cot_reasoning += "\\nNow examining the auxiliary perspective (Image B):\\n"
            visible_b = [(obj_name, bbox) for obj_name, bbox in bbox_b.items() if bbox is not None]
            
            for i, (obj_name, bbox) in enumerate(visible_b):
                pattern = sentence_patterns[i % len(sentence_patterns)]
                center = get_bbox_center(bbox)
                cot_reasoning += f"- {pattern.format(obj_type=obj_name, bbox=bbox)} with center at [{center[0]:.1f}, {center[1]:.1f}]\\n"
            
            # Sort objects by position in auxiliary perspective
            if visible_b:
                sorted_objects_b = sort_objects_by_position({name: bbox for name, bbox in visible_b}, "left_to_right")
                if len(sorted_objects_b) > 1:
                    cot_reasoning += "\\nSpatial arrangement from left to right in auxiliary perspective:\\n"
                    for pos, (obj_name, bbox) in enumerate(sorted_objects_b, 1):
                        center = get_bbox_center(bbox)
                        cot_reasoning += f"- Position {pos}: {obj_name} at center [{center[0]:.1f}, {center[1]:.1f}]\\n"
            
            # Analyze which perspective shows the target object and determine grasping position
            cot_reasoning += "\\nAnalyzing target object for grasping:\\n"
            
            if target_perspective == "main":
                cot_reasoning += f"The target {target_object} is visible in the main perspective at {target_bbox}.\\n"
                # Analyze position within the main perspective
                main_sorted = sort_objects_by_position(bbox_a, "left_to_right")
                for pos, (obj_name, bbox) in enumerate(main_sorted, 1):
                    if obj_name == target_object:
                        cot_reasoning += f"In the main perspective, {target_object} is in position {pos} from the left.\\n"
                        break
            elif target_perspective == "auxiliary":
                cot_reasoning += f"The target {target_object} is visible in the auxiliary perspective at {target_bbox}.\\n"
                # Analyze position within the auxiliary perspective
                aux_sorted = sort_objects_by_position(bbox_b, "left_to_right")
                for pos, (obj_name, bbox) in enumerate(aux_sorted, 1):
                    if obj_name == target_object:
                        cot_reasoning += f"In the auxiliary perspective, {target_object} is in position {pos} from the left.\\n"
                        break
            
            # Verify the position matches the instruction
            if "second position from the left" in grasping_instruction.lower():
                cot_reasoning += f"The instruction asks for the object in the second position from the left, which is {target_object}.\\n"
            elif "first position from the left" in grasping_instruction.lower() or "leftmost" in grasping_instruction.lower():
                cot_reasoning += f"The instruction asks for the leftmost object, which is {target_object}.\\n"
            elif "third position from the left" in grasping_instruction.lower():
                cot_reasoning += f"The instruction asks for the object in the third position from the left, which is {target_object}.\\n"
            
            cot_reasoning += f"\\nFor grasping, I need to determine the center coordinates of {target_object}.\\n"
            cot_reasoning += f"The {target_object} is located at bounding box {target_bbox}.\\n"
            cot_reasoning += f"The center coordinates are: [{target_center[0]:.1f}, {target_center[1]:.1f}].\\n"
            
            # Determine which image contains the target for level 3 format
            if target_perspective == "main":
                image_number = 0
                cot_reasoning += f"\\nThe target object is in the main perspective (image 0), so I should grasp at image 0 coordinates [{target_center[0]:.1f}, {target_center[1]:.1f}]."
            else:  # auxiliary
                image_number = 1
                cot_reasoning += f"\\nThe target object is in the auxiliary perspective (image 1), so I should grasp at image 1 coordinates [{target_center[0]:.1f}, {target_center[1]:.1f}]."
            
            ground_truth = f"<think>\\n{cot_reasoning}\\n</think>\\n\\n\\\\boxed{{{image_number}, [{target_center[0]:.1f}, {target_center[1]:.1f}]}}"
            
            # Structure the data following VERL format
            # Use different data_source for train/test to enable different reward functions
            data_source_suffix = "_train" if split == "train" else "_val"
            data_source_name = f"corl_l3_{dataset_name.lower()}{data_source_suffix}"
            
            processed_item = {
                "images": [
                    {
                        "bytes": image_binaries[0],
                        "path": None
                    },
                    {
                        "bytes": image_binaries[1], 
                        "path": None
                    }
                ],
                # Use different data_source for train/test to enable different reward functions
                "data_source": data_source_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": instruction_following
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "ability": "spatial_reasoning_grasping",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "metadata": {
                    "task_name": "object_grasping",
                    "task_id": gt_info.get('task_id', f'{dataset_name}_{idx}'),
                    "layout_id": gt_info.get('layout_id', -1),
                    "robots": gt_info.get('robots', []),
                    "objects": gt_info.get('objects', []),
                    "init_pos": gt_info.get('init_pos', {}),
                    "target_object": target_object,
                    "target_center": target_center,
                    "original_question": original_question,
                    "grasping_instruction": grasping_instruction
                }
            }
            processed_data.append(processed_item)
            
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {str(e)}")
            skipped_count += 1
            continue
    
    logger.info(f"Processed {len(processed_data)} samples for {split} split from {dataset_name}, skipped {skipped_count} samples")
    return processed_data


def process_single_dataset_for_merge(dataset_name, base_dir, args):
    """Process a single Spatial_Understand dataset and return the processed data for merging"""
    logger.info(f"Processing dataset for merge: {dataset_name}")
    
    # Dataset paths
    dataset_dir = os.path.join(base_dir, dataset_name)
    meta_data_file = os.path.join(dataset_dir, "meta_data.json")
    
    if not os.path.exists(meta_data_file):
        logger.error(f"Meta data file not found: {meta_data_file}")
        return None, None
    
    # Load metadata
    meta_data = load_json_data(meta_data_file)
    logger.info(f"Loaded {len(meta_data)} samples from {dataset_name}")
    
    # If requested limit samples
    if args.max_samples:
        meta_data = meta_data[:args.max_samples]
        logger.info(f"Limited to {len(meta_data)} samples for testing")
    
    # Split for train/test (80/20)
    split_idx = int(len(meta_data) * 0.8)
    train_data = meta_data[:split_idx]
    test_data = meta_data[split_idx:]
    logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Use the same process_data function
    def process_data_with_dataset_info(data, split):
        """Process data and add dataset information"""
        processed_data = process_data_generic(data, split, dataset_name, dataset_dir, args)
        # Add dataset origin info to metadata for merged datasets
        for item in processed_data:
            item['metadata']['source_dataset'] = dataset_name
        return processed_data
    
    # Process train and test data
    train_processed = process_data_with_dataset_info(train_data, 'train')
    test_processed = process_data_with_dataset_info(test_data, 'test')
    
    if not train_processed or not test_processed:
        logger.error(f"No samples were successfully processed for {dataset_name}!")
        return None, None
    
    logger.info(f" {dataset_name} processed: {len(train_processed)} train, {len(test_processed)} test")
    return train_processed, test_processed


def process_single_dataset(dataset_name, base_dir, output_base_dir, args):
    """Process a single Spatial_Understand dataset"""
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Dataset paths
    dataset_dir = os.path.join(base_dir, dataset_name)
    meta_data_file = os.path.join(dataset_dir, "meta_data.json")
    
    if not os.path.exists(meta_data_file):
        logger.error(f"Meta data file not found: {meta_data_file}")
        return False
    
    # Load metadata
    meta_data = load_json_data(meta_data_file)
    logger.info(f"Loaded {len(meta_data)} samples from {dataset_name}")
    
    # If requested limit samples
    if args.max_samples:
        meta_data = meta_data[:args.max_samples]
        logger.info(f"Limited to {len(meta_data)} samples for testing")
    
    # Split for train/test (80/20)
    split_idx = int(len(meta_data) * 0.8)
    train_data = meta_data[:split_idx]
    test_data = meta_data[split_idx:]
    logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Process train and test data using the generic function
    train_processed = process_data_generic(train_data, 'train', dataset_name, dataset_dir, args)
    test_processed = process_data_generic(test_data, 'test', dataset_name, dataset_dir, args)
    
    if not train_processed:
        logger.error(f"No training samples were successfully processed for {dataset_name}!")
        return False
    if not test_processed:
        logger.error(f"No test samples were successfully processed for {dataset_name}!")
        return False
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_processed)
    test_dataset = Dataset.from_list(test_processed)
    
    logger.info(f"Created datasets with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Create output directory for this dataset
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet format
    logger.info(f"Saving {dataset_name} datasets to parquet format...")
    train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir, 'test.parquet'))
    
    logger.info(f"Saved {dataset_name} parquet files to {output_dir}")
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "data_source_train": f"corl_l3_spatial_understand_train",
        "data_source_val": f"corl_l3_spatial_understand_val",
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        "image_format": "compressed_jpeg" if args.compress_images else "original_binary",
        "images_per_sample": 2,
        "ability": "spatial_reasoning_grasping",
        "description": f"CORL-L3 {dataset_name} dual-image object grasping dataset for RL training",
        "compression_settings": {
            "enabled": args.compress_images,
            "jpeg_quality": args.image_quality if args.compress_images else None,
            "max_image_size": args.max_image_size if args.compress_images else None
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f" {dataset_name} processing complete!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch preprocess CORL-L3 Spatial_Understand datasets for VERL training')
    parser.add_argument(
        '--base_dir', default='./data/CORL-L3')
    parser.add_argument(
        '--output_dir', default='./data/CORL-L3/CORL-L3-verl')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per dataset for testing')
    parser.add_argument('--compress_images', action='store_true',
                        help='Compress images to reduce file size')
    parser.add_argument('--image_quality', type=int, default=70,
                        help='JPEG compression quality (1-100)')
    parser.add_argument('--max_image_size', type=int, default=384,
                        help='Maximum image dimension')
    parser.add_argument('--datasets', nargs='+', 
                        default=['Spatial_Understand_0','Spatial_Understand_1', 'Spatial_Understand_2', 'Spatial_Understand_3', 'Spatial_Understand_4', 'Spatial_Understand_5'],
                        help='List of datasets to process')
    parser.add_argument('--merge_all', action='store_true',
                        help='Merge all Spatial_Understand datasets into single train/test files')
    args = parser.parse_args()
    
    # Create output base directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    processed_datasets = []
    failed_datasets = []
    all_train_data = []
    all_test_data = []
    
    for dataset_name in args.datasets:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Starting processing: {dataset_name}")
        logger.info(f"{'='*60}")
        
        if args.merge_all:
            # Collect processed data instead of saving immediately
            train_data, test_data = process_single_dataset_for_merge(dataset_name, args.base_dir, args)
            if train_data is not None and test_data is not None:
                all_train_data.extend(train_data)
                all_test_data.extend(test_data)
                processed_datasets.append(dataset_name)
            else:
                failed_datasets.append(dataset_name)
        else:
            # Original individual processing
            success = process_single_dataset(dataset_name, args.base_dir, args.output_dir, args)
            if success:
                processed_datasets.append(dataset_name)
            else:
                failed_datasets.append(dataset_name)
    
    # Handle merged dataset saving
    if args.merge_all and all_train_data and all_test_data:
        logger.info(f"\\n{'='*60}")
        logger.info("Creating merged dataset...")
        logger.info(f"{'='*60}")
        
        # Convert to HuggingFace datasets
        merged_train_dataset = Dataset.from_list(all_train_data)
        merged_test_dataset = Dataset.from_list(all_test_data)
        
        logger.info(f"Created merged datasets with {len(merged_train_dataset)} training samples and {len(merged_test_dataset)} test samples")
        
        # Save merged dataset
        merged_output_dir = os.path.join(args.output_dir, 'CORL_L3_Merged')
        os.makedirs(merged_output_dir, exist_ok=True)
        
        logger.info("Saving merged datasets to parquet format...")
        merged_train_dataset.to_parquet(os.path.join(merged_output_dir, 'train.parquet'))
        merged_test_dataset.to_parquet(os.path.join(merged_output_dir, 'test.parquet'))
        
        # Save merged metadata
        merged_metadata = {
            "dataset_name": "CORL_L3_Merged",
            "data_source_train": "corl_l3_merged_train",
            "data_source_val": "corl_l3_merged_val", 
            "total_train_samples": len(merged_train_dataset),
            "total_test_samples": len(merged_test_dataset),
            "source_datasets": processed_datasets,
            "image_format": "compressed_jpeg" if args.compress_images else "original_binary",
            "images_per_sample": 2,
            "ability": "spatial_reasoning_grasping",
            "description": "CORL-L3 merged object grasping dataset for RL training",
            "compression_settings": {
                "enabled": args.compress_images,
                "jpeg_quality": args.image_quality if args.compress_images else None,
                "max_image_size": args.max_image_size if args.compress_images else None
            }
        }
        
        with open(os.path.join(merged_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Merged dataset saved to {merged_output_dir}")
    
    # Final summary
    logger.info(f"\\n{'='*80}")
    logger.info("<� BATCH PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f" Successfully processed: {len(processed_datasets)} datasets")
    for ds in processed_datasets:
        logger.info(f"   - {ds}")
    
    if failed_datasets:
        logger.warning(f"L Failed to process: {len(failed_datasets)} datasets")
        for ds in failed_datasets:
            logger.warning(f"   - {ds}")
    
    logger.info(f"=� Output location: {args.output_dir}")
    logger.info("="*80)
    
    # Print verification commands
    print(f"\\n= Verification commands:")
    if args.merge_all and all_train_data:
        print(f"python -c \"import datasets; ds=datasets.Dataset.from_parquet('{args.output_dir}/CORL_L3_Merged/train.parquet'); print('CORL_L3_Merged:', ds.num_rows, 'samples')\"")
    else:
        for ds in processed_datasets:
            print(f"python -c \"import datasets; ds=datasets.Dataset.from_parquet('{args.output_dir}/{ds}/train.parquet'); print('{ds}:', ds.num_rows, 'samples')\"")