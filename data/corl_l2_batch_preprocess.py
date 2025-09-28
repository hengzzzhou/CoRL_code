"""
CORL-L2 Dataset Batch Preprocessing Script - Spatial Understanding Tasks
Adapted from corl_l1_batch_preprocess.py for processing multiple Spatial_Understand datasets
Processes Spatial_Understand_0 through Spatial_Understand_5 datasets for VERL training
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
            # bbox格式: [x1, y1, x2, y2]
            scaled_bbox[obj_name] = [
                int(bbox[0] * scale_ratio),  # x1
                int(bbox[1] * scale_ratio),  # y1
                int(bbox[2] * scale_ratio),  # x2
                int(bbox[3] * scale_ratio)   # y2
            ]
        else:
            scaled_bbox[obj_name] = None
    return scaled_bbox


def create_bbox_context(bbox_dict, image_name):
    """Create context description based on bounding box information"""
    context_lines = []
    for obj_name, bbox in bbox_dict.items():
        if bbox is not None:
            # bbox格式: [x1, y1, x2, y2]
            context_lines.append(
                f"{obj_name} detected in {image_name} at coordinates {bbox}")
        else:
            context_lines.append(f"{obj_name} not visible in {image_name}")
    return context_lines


def calculate_spatial_relationships(bbox_dict):
    """Calculate spatial relationships between objects for spatial reasoning CoT"""
    relationships = []
    visible_objects = {obj_name: bbox for obj_name, bbox in bbox_dict.items() if bbox is not None}
    
    if len(visible_objects) < 2:
        return relationships
    
    obj_list = list(visible_objects.items())
    for i in range(len(obj_list)):
        for j in range(i+1, len(obj_list)):
            obj1_name, bbox1 = obj_list[i]
            obj2_name, bbox2 = obj_list[j]
            
            # Calculate centers
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center1_y = (bbox1[1] + bbox1[3]) / 2
            center2_x = (bbox2[0] + bbox2[2]) / 2
            center2_y = (bbox2[1] + bbox2[3]) / 2
            
            # Calculate distance
            distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
            
            # Determine relative positions
            if center1_x < center2_x - 50:  # threshold for "left of"
                if center1_y < center2_y - 50:
                    relationship = f"{obj1_name} is to the upper-left of {obj2_name}"
                elif center1_y > center2_y + 50:
                    relationship = f"{obj1_name} is to the lower-left of {obj2_name}"
                else:
                    relationship = f"{obj1_name} is to the left of {obj2_name}"
            elif center1_x > center2_x + 50:
                if center1_y < center2_y - 50:
                    relationship = f"{obj1_name} is to the upper-right of {obj2_name}"
                elif center1_y > center2_y + 50:
                    relationship = f"{obj1_name} is to the lower-right of {obj2_name}"
                else:
                    relationship = f"{obj1_name} is to the right of {obj2_name}"
            else:  # roughly same x position
                if center1_y < center2_y - 50:
                    relationship = f"{obj1_name} is above {obj2_name}"
                elif center1_y > center2_y + 50:
                    relationship = f"{obj1_name} is below {obj2_name}"
                else:
                    relationship = f"{obj1_name} is near {obj2_name}"
            
            relationships.append((relationship, distance))
    
    return relationships


def process_data_generic(data, split, dataset_name, dataset_dir, args):
    """Generic data processing function that can be used for both individual and merged processing"""
    # Spatial reasoning task system prompt - adapted for spatial understanding
    instruction_following = (
        "You are analyzing two images from different robot perspectives viewing the same environment. "
        "These images represent two distinct robot viewpoints that may show overlapping or complementary "
        "parts of the scene. FIRST analyze the spatial relationships and objects visible from each robot's "
        "perspective, understanding how the different viewpoints relate to each other. Think through the "
        "reasoning process as an internal monologue, considering what each robot can see and how their "
        "observations combine to solve the spatial reasoning problem. "
        "When identifying objects RELEVANT TO THE QUESTION, you must provide their bounding box coordinates "
        "in the format: [x1, y1, x2, y2] where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. "
        "For example: 'I can see a cup at bounding box [100, 150, 200, 250]'. "
        "Only provide bounding boxes for objects that are directly related to answering the question. "
        "For spatial reasoning questions, analyze the relative positions, distances, and spatial relationships "
        "between objects to determine the correct answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer should be the object name and put in \\boxed{}."
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
            
            question = gt_info['description']
            answer = gt_info['ground_truth']
            
            # Build user message (dual image format)
            user_message = f"main picture:<image>\\nauxiliary picture:<image>\\nQuestion: {question}"
            
            # Build detailed chain-of-thought reasoning with spatial analysis
            cot_reasoning = f"I need to analyze images from two robot perspectives to answer the spatial reasoning question: '{question}'.\\n\\n"
            
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
                cot_reasoning += f"- {pattern.format(obj_type=obj_name, bbox=bbox)}\\n"
            
            # Calculate spatial relationships within Image A
            relationships_a = calculate_spatial_relationships(bbox_a)
            if relationships_a:
                cot_reasoning += "\\nSpatial relationships in the main perspective:\\n"
                for relationship, distance in relationships_a:
                    cot_reasoning += f"- {relationship} (distance: {distance:.1f} pixels)\\n"
            
            cot_reasoning += "\\nNow examining the auxiliary perspective (Image B):\\n"
            visible_b = [(obj_name, bbox) for obj_name, bbox in bbox_b.items() if bbox is not None]
            
            for i, (obj_name, bbox) in enumerate(visible_b):
                pattern = sentence_patterns[i % len(sentence_patterns)]
                cot_reasoning += f"- {pattern.format(obj_type=obj_name, bbox=bbox)}\\n"
            
            # Calculate spatial relationships within Image B
            relationships_b = calculate_spatial_relationships(bbox_b)
            if relationships_b:
                cot_reasoning += "\\nSpatial relationships in the auxiliary perspective:\\n"
                for relationship, distance in relationships_b:
                    cot_reasoning += f"- {relationship} (distance: {distance:.1f} pixels)\\n"
            
            # Add cross-perspective analysis
            cot_reasoning += "\\nAnalyzing spatial correspondence between the two perspectives:\\n"
            
            # Combine all visible objects from both perspectives
            all_objects = set(bbox_a.keys()) | set(bbox_b.keys())
            object_visibility = {}
            
            for obj_name in all_objects:
                in_a = bbox_a.get(obj_name) is not None
                in_b = bbox_b.get(obj_name) is not None
                
                if in_a and in_b:
                    cot_reasoning += f"- {obj_name} is visible in both perspectives at {bbox_a[obj_name]} and {bbox_b[obj_name]} respectively\\n"
                    object_visibility[obj_name] = 'both'
                elif in_a:
                    cot_reasoning += f"- {obj_name} is only visible in main perspective at {bbox_a[obj_name]}\\n"
                    object_visibility[obj_name] = 'main_only'
                elif in_b:
                    cot_reasoning += f"- {obj_name} is only visible in auxiliary perspective at {bbox_b[obj_name]}\\n"
                    object_visibility[obj_name] = 'aux_only'
            
            # Spatial reasoning logic based on the question type
            if "farthest from" in question.lower():
                # Extract the reference object from the question
                reference_obj = None
                for obj_name in all_objects:
                    if obj_name in question.lower():
                        reference_obj = obj_name
                        break
                
                if reference_obj:
                    cot_reasoning += f"\\nTo answer '{question}', I need to find which object is farthest from {reference_obj}.\\n"
                    
                    # Calculate distances from reference object
                    max_distance = 0
                    farthest_obj = None
                    
                    # Get reference object position (prioritize perspective where it's visible)
                    ref_bbox = None
                    ref_perspective = None
                    if bbox_a.get(reference_obj) is not None:
                        ref_bbox = bbox_a[reference_obj]
                        ref_perspective = "main"
                    elif bbox_b.get(reference_obj) is not None:
                        ref_bbox = bbox_b[reference_obj]
                        ref_perspective = "auxiliary"
                    
                    if ref_bbox:
                        ref_center_x = (ref_bbox[0] + ref_bbox[2]) / 2
                        ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2
                        
                        # Check distances in the same perspective first
                        current_bbox_dict = bbox_a if ref_perspective == "main" else bbox_b
                        
                        for obj_name, obj_bbox in current_bbox_dict.items():
                            if obj_name != reference_obj and obj_bbox is not None:
                                obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
                                obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2
                                distance = ((ref_center_x - obj_center_x)**2 + (ref_center_y - obj_center_y)**2)**0.5
                                
                                cot_reasoning += f"- Distance from {reference_obj} to {obj_name}: {distance:.1f} pixels\\n"
                                
                                if distance > max_distance:
                                    max_distance = distance
                                    farthest_obj = obj_name
                        
                        # Also check objects only visible in the other perspective
                        other_bbox_dict = bbox_b if ref_perspective == "main" else bbox_a
                        for obj_name, obj_bbox in other_bbox_dict.items():
                            if obj_name != reference_obj and obj_bbox is not None and current_bbox_dict.get(obj_name) is None:
                                # For cross-perspective comparison, assume maximum possible distance
                                cot_reasoning += f"- {obj_name} is only visible in the other perspective, likely at maximum distance from {reference_obj}\\n"
                                # Give it a high distance score
                                if max_distance < 1000:  # If no close objects found
                                    max_distance = 1500  # Assume far distance
                                    farthest_obj = obj_name
                    
                    if farthest_obj:
                        cot_reasoning += f"\\nBased on the spatial analysis, {farthest_obj} is located farthest from {reference_obj}."
                    
            elif "closest to" in question.lower() or "nearest to" in question.lower():
                # Similar logic but finding minimum distance
                reference_obj = None
                for obj_name in all_objects:
                    if obj_name in question.lower():
                        reference_obj = obj_name
                        break
                
                if reference_obj:
                    cot_reasoning += f"\\nTo answer '{question}', I need to find which object is closest to {reference_obj}.\\n"
                    # Add similar distance calculation logic for closest object
                    
            cot_reasoning += f"\\nTherefore, the answer to '{question}' is {answer}."
            
            ground_truth = f"<think>\\n{cot_reasoning}\\n</think>\\n\\n\\\\boxed{{{answer}}}"
            
            # Structure the data following VERL format
            # Use different data_source for train/test to enable different reward functions
            data_source_suffix = "_train" if split == "train" else "_val"
            data_source_name = f"corl_l2_{dataset_name.lower()}{data_source_suffix}"
            
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
                "ability": "spatial_reasoning_understanding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "metadata": {
                    "task_name": "spatial_understanding",
                    "task_id": gt_info.get('task_id', f'{dataset_name}_{idx}'),
                    "layout_id": gt_info.get('layout_id', -1),
                    "robots": gt_info.get('robots', []),
                    "objects": gt_info.get('objects', []),
                    "init_pos": gt_info.get('init_pos', {})
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
    
    logger.info(f"✓ {dataset_name} processed: {len(train_processed)} train, {len(test_processed)} test")
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
        "data_source_train": f"corl_l2_{dataset_name.lower()}_train",
        "data_source_val": f"corl_l2_{dataset_name.lower()}_val",
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        "image_format": "compressed_jpeg" if args.compress_images else "original_binary",
        "images_per_sample": 2,
        "ability": "spatial_reasoning_understanding",
        "description": f"CORL-L2 {dataset_name} dual-image spatial understanding dataset for RL training",
        "compression_settings": {
            "enabled": args.compress_images,
            "jpeg_quality": args.image_quality if args.compress_images else None,
            "max_image_size": args.max_image_size if args.compress_images else None
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ {dataset_name} processing complete!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch preprocess CORL-L2 Spatial_Understand datasets for VERL training')
    parser.add_argument(
        '--base_dir', default='./data/CORL-L2')
    parser.add_argument(
        '--output_dir', default='./data/CORL-L2/CORL-L2-verl')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per dataset for testing')
    parser.add_argument('--compress_images', action='store_true',default='true',
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
        merged_output_dir = os.path.join(args.output_dir, 'CORL_L2_Merged')
        os.makedirs(merged_output_dir, exist_ok=True)
        
        logger.info("Saving merged datasets to parquet format...")
        merged_train_dataset.to_parquet(os.path.join(merged_output_dir, 'train.parquet'))
        merged_test_dataset.to_parquet(os.path.join(merged_output_dir, 'test.parquet'))
        
        # Save merged metadata
        merged_metadata = {
            "dataset_name": "CORL_L2_Merged",
            "data_source_train": "corl_l2_merged_train",
            "data_source_val": "corl_l2_merged_val", 
            "total_train_samples": len(merged_train_dataset),
            "total_test_samples": len(merged_test_dataset),
            "source_datasets": processed_datasets,
            "image_format": "compressed_jpeg" if args.compress_images else "original_binary",
            "images_per_sample": 2,
            "ability": "spatial_reasoning_understanding",
            "description": "CORL-L2 merged spatial understanding dataset for RL training",
            "compression_settings": {
                "enabled": args.compress_images,
                "jpeg_quality": args.image_quality if args.compress_images else None,
                "max_image_size": args.max_image_size if args.compress_images else None
            }
        }
        
        with open(os.path.join(merged_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Merged dataset saved to {merged_output_dir}")
    
    # Final summary
    logger.info(f"\\n{'='*80}")
    logger.info("🏁 BATCH PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully processed: {len(processed_datasets)} datasets")
    for ds in processed_datasets:
        logger.info(f"   - {ds}")
    
    if failed_datasets:
        logger.warning(f"❌ Failed to process: {len(failed_datasets)} datasets")
        for ds in failed_datasets:
            logger.warning(f"   - {ds}")
    
    logger.info(f"📁 Output location: {args.output_dir}")
    logger.info("="*80)
    
    # Print verification commands
    print(f"\\n🔍 Verification commands:")
    if args.merge_all and all_train_data:
        print(f"python -c \"import datasets; ds=datasets.Dataset.from_parquet('{args.output_dir}/CORL_L2_Merged/train.parquet'); print('CORL_L2_Merged:', ds.num_rows, 'samples')\"")
    else:
        for ds in processed_datasets:
            print(f"python -c \"import datasets; ds=datasets.Dataset.from_parquet('{args.output_dir}/{ds}/train.parquet'); print('{ds}:', ds.num_rows, 'samples')\"")