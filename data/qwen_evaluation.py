#!/usr/bin/env python3
"""
GPT-5 Model Evaluation Script for CORL Datasets

This script evaluates GPT-5 model performance on two parquet datasets:
1. merged_test_data.parquet
2. test_where2place (1).parquet

The evaluation uses data_source-specific reward functions from:
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Add the reward score module to Python path
sys.path.append('./verl')

try:
    from verl.utils.reward_score import default_compute_score
except ImportError as e:
    print(f"Error importing reward score module: {e}")
    print("Please ensure the path is correct and the module is available")
    sys.exit(1)

# Logger will be configured later with model-specific filename
logger = logging.getLogger(__name__)


class VLLMEvaluator:
    """VLLM Model Evaluator for CORL datasets using trained checkpoint"""

    def __init__(self, dataset_paths: List[str], output_dir: str = "./evaluation_results",
                 use_real_api: bool = True, model_name: str = "qwen-vl-trained", target_data_sources: Optional[List[str]] = None,
                 api_base: str = "http://127.0.0.1:8000/v1/", api_key: str = "EMPTY", processor: Optional[Any] = None):
        """
        Initialize the evaluator

        Args:
            dataset_paths: List of paths to parquet datasets
            output_dir: Directory to save evaluation results
            use_real_api: Whether to use real API or mock responses
            model_name: Model name to use for API calls (should match VLLM served model name)
            target_data_sources: List of specific data sources to evaluate (None for all)
            api_base: Base URL for VLLM server API
            api_key: API key for VLLM server (usually "EMPTY" for local server)
            processor: Processor used during training (for consistent image processing)
        """
        self.dataset_paths = dataset_paths
        self.use_real_api = use_real_api
        self.model_name = model_name
        self.target_data_sources = target_data_sources
        self.api_base = api_base
        self.api_key = api_key
        self.processor = processor

        # Create model-specific output directory
        self.output_dir = Path(
            output_dir) / self.model_name.replace("/", "_").replace(":", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging with model-specific log file
        clean_model_name = self.model_name.replace(
            "/", "_").replace(":", "_").replace("-", "_")
        log_file = self.output_dir / f"{clean_model_name}_evaluation.log"

        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )

        # Initialize OpenAI client for VLLM server
        if self.use_real_api:
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
        else:
            self.client = None

        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'data_source_counts': {},
            'data_source_scores': {},
            'evaluation_errors': []
        }

        logger.info(
            f"Initialized VLLMEvaluator with {len(dataset_paths)} datasets")
        logger.info(f"Output directory: {self.output_dir}")
        if self.target_data_sources:
            logger.info(f"Target data sources: {', '.join(self.target_data_sources)}")
        if self.use_real_api:
            logger.info(f"Model: {self.model_name}")
            logger.info(f"API Base: {self.api_base}")
            logger.info("VLLM client initialized")

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and validate a parquet dataset"""
        try:
            logger.info(f"Loading dataset: {dataset_path}")
            df = pd.read_parquet(dataset_path)
            logger.info(
                f"Loaded dataset with {len(df)} samples and columns: {list(df.columns)}")

            # Validate required columns
            required_columns = ['data_source']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(
                    f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            raise

    def get_data_source_info(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get information about data sources in the dataset"""
        data_source_counts = df['data_source'].value_counts().to_dict()
        logger.info("Data source distribution:")
        for source, count in data_source_counts.items():
            logger.info(f"  {source}: {count} samples")
        return data_source_counts

    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string for API"""
        return base64.b64encode(image_bytes).decode('utf-8')

    def prepare_images_for_api(self, images_data: Any) -> Dict[str, Any]:
        """
        Prepare images for VLLM API format and extract dimensions
        
        For VLLM serving with trained checkpoint, we need to ensure the image processing
        is consistent with training. This uses the same process_image function as training.
        """
        image_messages = []
        image_dimensions = []

        try:
            # Import VERL's vision utils for consistent processing - same as training
            sys.path.append('./verl')
            from verl.utils.dataset.vision_utils import process_image
            
            # Check if we have any images data
            if images_data is None:
                return {"messages": image_messages, "dimensions": image_dimensions}

            # Handle numpy array case
            if isinstance(images_data, np.ndarray):
                # Check if the array has elements
                if images_data.size == 0:
                    return {"messages": image_messages, "dimensions": image_dimensions}

                # Process each image in the array using the EXACT same process as training
                for img_data in images_data:
                    if isinstance(img_data, dict):
                        try:
                            # Step 1: Same as training - use process_image
                            processed_image = process_image(img_data)
                            
                            # Step 2: Apply processor if available (same as training)
                            if self.processor is not None:
                                # Process with the same processor as training
                                processor_output = self.processor(text=[""], images=[processed_image], return_tensors="pt")
                                
                                # For VLLM API, we still need base64, so convert back to PIL
                                # This ensures the image went through the exact same processing as training
                                pixel_values = processor_output.get('pixel_values')
                                if pixel_values is not None:
                                    # Convert tensor back to PIL for base64 encoding
                                    # Note: This is a workaround - ideally VLLM would accept the processed tensors directly
                                    final_image = processed_image  # Use the processed PIL image
                                else:
                                    final_image = processed_image
                            else:
                                final_image = processed_image
                            
                            # Get dimensions
                            width, height = final_image.size
                            image_dimensions.append({"width": width, "height": height})
                            
                            # Convert to base64 with higher quality to minimize compression artifacts
                            buffer = BytesIO()
                            final_image.save(buffer, format='PNG')  # Use PNG to avoid JPEG compression
                            image_bytes = buffer.getvalue()
                            base64_image = self.encode_image_to_base64(image_bytes)
                            
                            image_messages.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            })
                        except Exception as process_error:
                            logger.warning(f"Failed to process image: {process_error}")
                            continue
                            
            # Handle regular list case
            elif isinstance(images_data, list) and len(images_data) > 0:
                for img_item in images_data:
                    if isinstance(img_item, dict):
                        try:
                            # Step 1: Same as training - use process_image
                            processed_image = process_image(img_item)
                            
                            # Step 2: Apply processor if available (same as training)
                            if self.processor is not None:
                                # Process with the same processor as training
                                processor_output = self.processor(text=[""], images=[processed_image], return_tensors="pt")
                                
                                # For VLLM API, we still need base64, so convert back to PIL
                                # This ensures the image went through the exact same processing as training
                                pixel_values = processor_output.get('pixel_values')
                                if pixel_values is not None:
                                    # Convert tensor back to PIL for base64 encoding
                                    # Note: This is a workaround - ideally VLLM would accept the processed tensors directly
                                    final_image = processed_image  # Use the processed PIL image
                                else:
                                    final_image = processed_image
                            else:
                                final_image = processed_image
                            
                            # Get dimensions
                            width, height = final_image.size
                            image_dimensions.append({"width": width, "height": height})
                            
                            # Convert to base64 with higher quality to minimize compression artifacts
                            buffer = BytesIO()
                            final_image.save(buffer, format='PNG')  # Use PNG to avoid JPEG compression
                            image_bytes = buffer.getvalue()
                            base64_image = self.encode_image_to_base64(image_bytes)
                            
                            image_messages.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            })
                        except Exception as process_error:
                            logger.warning(f"Failed to process image: {process_error}")
                            continue
                            
        except Exception as e:
            logger.warning(f"Failed to prepare images: {e}")

        return {"messages": image_messages, "dimensions": image_dimensions}

    def call_vllm_api(self, sample: Dict[str, Any], max_retries: int = 10) -> str:
        """
        Call VLLM API with the sample data

        Args:
            sample: Dictionary containing sample data including images and prompt
            max_retries: Maximum number of retry attempts

        Returns:
            VLLM model response string
        """

        for attempt in range(max_retries):
            try:
                data_source = sample['data_source']
                original_prompt = sample.get('prompt', '')

                # Prepare images for API
                images_data = sample.get('images', [])
                logger.debug(f"Images data type: {type(images_data)}")

                image_result = self.prepare_images_for_api(images_data)
                image_messages = image_result["messages"]
                image_dimensions = image_result["dimensions"]

                # Create dimension text if images are present
                dimension_text = ""
                if image_dimensions:
                    dim_texts = []
                    for i, dim in enumerate(image_dimensions):
                        dim_texts.append(f"Image {i+1}: {dim['width']}x{dim['height']}")
                    dimension_text = f" [Image dimensions: {', '.join(dim_texts)}]"

                if not image_messages:
                    logger.warning(
                        "No images found in sample, using text-only prompt")

                # Parse original prompt to extract system and user messages
                messages = []

                if isinstance(original_prompt, np.ndarray) and original_prompt.size > 0:
                    # Handle numpy array of message dictionaries
                    for msg in original_prompt:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            if msg['role'] == 'system':
                                messages.append({
                                    "role": "system",
                                    "content": msg['content']
                                })
                            elif msg['role'] == 'user':
                                # For user messages, add images if available
                                text_content = msg['content'] + dimension_text
                                content_parts = [
                                    {"type": "text", "text": text_content}]
                                if image_messages:
                                    content_parts.extend(image_messages)
                                messages.append({
                                    "role": "user",
                                    "content": content_parts
                                })
                else:
                    # Fallback for string prompts
                    text_content = str(original_prompt) + dimension_text
                    content_parts = [
                        {"type": "text", "text": text_content}]
                    if image_messages:
                        content_parts.extend(image_messages)
                    messages.append({
                        "role": "user",
                        "content": content_parts
                    })

                # Add special handling for corl_l2_spatial_understand_val
                if data_source == 'corl_l2_spatial_understand_val':
                    if messages and len(messages) > 0:
                        # Find the user message and add the choices constraint
                        for msg in messages:
                            if msg['role'] == 'user':
                                if isinstance(msg['content'], list):
                                    # Find the text content
                                    for content_item in msg['content']:
                                        if content_item.get('type') == 'text':
                                            content_item['text'] += " Your answer should be one of the following options: [\"apple\", \"banana\", \"bowl\", \"bread\", \"cardboardbox\", \"clothes_iron\", \"cup\", \"fork\", \"green_peas\", \"hammer\", \"kettle\", \"knife\", \"meat\", \"peach\", \"pear\", \"pizza\", \"plate\", \"pumpkin\", \"scissors\", \"shoe\", \"tomato\", \"towel\", \"wine\", \"wooden_cutting_board\"]. Format your final answer as \\\\boxed\\{answer\\}."
                                            break
                                elif isinstance(msg['content'], str):
                                    msg['content'] += " Your answer should be one of the following options: [\"apple\", \"banana\", \"bowl\", \"bread\", \"cardboardbox\", \"clothes_iron\", \"cup\", \"fork\", \"green_peas\", \"hammer\", \"kettle\", \"knife\", \"meat\", \"peach\", \"pear\", \"pizza\", \"plate\", \"pumpkin\", \"scissors\", \"shoe\", \"tomato\", \"towel\", \"wine\", \"wooden_cutting_board\"]. <think> ...</think>\n\n\\\\boxed{}"
                                break
                if data_source == 'where2place_point_test':
                    if messages and len(messages) > 0:
                        # Find the user message and add the choices constraint
                        for msg in messages:
                            if msg['role'] == 'user':
                                if isinstance(msg['content'], list):
                                    # Find the text content
                                    for content_item in msg['content']:
                                        if content_item.get('type') == 'text':
                                            content_item['text'] += " Format your final answer as \\\\boxed\\{answer\\}."
                                            break
                                elif isinstance(msg['content'], str):
                                    msg['content'] += "<think> ...</think>\n\n\\\\boxed{}"
                                break
                # if data_source == 'corl_l3_spatial_understand_val':
                #     if messages and len(messages) > 0:
                #         # Find the user message and add the choices constraint
                #         for msg in messages:
                #             if msg['role'] == 'user':
                #                 if isinstance(msg['content'], list):
                #                     # Find the text content
                #                     for content_item in msg['content']:
                #                         if content_item.get('type') == 'text':
                #                             content_item['text'] += " The size of each image is 384*216."
                #                             break

                # Call OpenAI API with timeout
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(
                    f"GPT API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)  # Reduced wait time
                else:
                    logger.error(
                        f"All VLLM API attempts failed, falling back to mock response")
                    return f"API call failed after {max_retries} attempts. Error: {str(e)}"

    def evaluate_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single sample

        Args:
            sample: Dictionary containing sample data
            sample_idx: Index of the sample for tracking

        Returns:
            Dictionary with evaluation results
        """
        try:
            data_source = sample['data_source']

            # Generate VLLM response using real API
            vllm_response = self.call_vllm_api(sample)

            # Extract ground truth if available
            ground_truth = sample['reward_model']['ground_truth']
            # Calculate score using the appropriate reward function
            try:
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=vllm_response,
                    ground_truth=ground_truth
                )

                # Handle different score return types
                if isinstance(score, dict):
                    # If score is a dictionary, extract the main score
                    main_score = score.get('score', score.get('reward', 0.0))
                    detailed_scores = score
                else:
                    main_score = float(score)
                    detailed_scores = {'score': main_score}

                return {
                    'sample_idx': sample_idx,
                    'data_source': data_source,
                    'vllm_response': vllm_response,
                    'ground_truth': ground_truth,
                    'main_score': main_score,
                    'detailed_scores': detailed_scores,
                    'evaluation_status': 'success',
                    'error': None
                }

            except Exception as score_error:
                logger.warning(
                    f"Score calculation failed for sample {sample_idx} with data_source {data_source}: {score_error}")
                return {
                    'sample_idx': sample_idx,
                    'data_source': data_source,
                    'vllm_response': vllm_response,
                    'ground_truth': ground_truth,
                    'main_score': 0.0,
                    'detailed_scores': {},
                    'evaluation_status': 'score_error',
                    'error': str(score_error)
                }

        except Exception as e:
            logger.error(f"Evaluation failed for sample {sample_idx}: {e}")
            return {
                'sample_idx': sample_idx,
                'data_source': sample.get('data_source', 'unknown'),
                'vllm_response': '',
                'ground_truth': '',
                'main_score': 0.0,
                'detailed_scores': {},
                'evaluation_status': 'error',
                'error': str(e)
            }

    def evaluate_dataset(self, df: pd.DataFrame, dataset_name: str,
                         max_samples: Optional[int] = None,
                         num_workers: int = 16) -> List[Dict[str, Any]]:
        """
        Evaluate all samples in a dataset

        Args:
            df: DataFrame containing the dataset
            dataset_name: Name of the dataset for logging
            max_samples: Maximum number of samples to evaluate (None for all)
            num_workers: Number of parallel workers

        Returns:
            List of evaluation results
        """
        logger.info(f"Starting evaluation of dataset: {dataset_name}")

        # Filter by target data sources if specified
        if self.target_data_sources:
            df = df[df['data_source'].isin(self.target_data_sources)].copy()
            logger.info(f"Filtered to {len(df)} samples with data_sources: {', '.join(self.target_data_sources)}")
            if len(df) == 0:
                logger.warning(f"No samples found for data_sources: {', '.join(self.target_data_sources)}")
                return []

        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples)
            logger.info(f"Limited evaluation to {max_samples} samples")

        # Convert to list of dictionaries for processing
        samples = df.to_dict('records')
        results = []

        # Evaluate samples with progress bar
        logger.info(f"Evaluating {len(samples)} samples...")

        if num_workers > 1:
            # Parallel evaluation with optimized thread pool
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix=f"eval-{dataset_name}"
            ) as executor:
                # Submit all tasks at once
                future_to_idx = {
                    executor.submit(self.evaluate_sample, sample, idx): idx
                    for idx, sample in enumerate(samples)
                }

                # Collect results as they complete
                completed_results = {}
                for future in tqdm(
                    concurrent.futures.as_completed(
                        future_to_idx, timeout=None),
                    total=len(samples),
                    desc=f"Evaluating {dataset_name}",
                    unit="samples"
                ):
                    idx = future_to_idx[future]
                    try:
                        # Short timeout for result retrieval
                        result = future.result(timeout=5)
                        completed_results[idx] = result
                    except Exception as e:
                        logger.error(
                            f"Failed to get result for sample {idx}: {e}")
                        # Create error result
                        completed_results[idx] = {
                            'sample_idx': idx,
                            'data_source': 'unknown',
                            'gpt5_response': '',
                            'ground_truth': '',
                            'main_score': 0.0,
                            'detailed_scores': {},
                            'evaluation_status': 'error',
                            'error': str(e)
                        }

                # Sort results by index to maintain order
                results = [completed_results[i] for i in range(len(samples))]
        else:
            # Sequential evaluation
            for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
                result = self.evaluate_sample(sample, idx)
                results.append(result)

        logger.info(f"Completed evaluation of {dataset_name}")
        return results

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation statistics"""
        total_samples = len(results)
        successful_evals = len(
            [r for r in results if r['evaluation_status'] == 'success'])
        failed_evals = total_samples - successful_evals

        # Calculate scores by data source
        data_source_stats = {}
        for result in results:
            data_source = result['data_source']
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    'count': 0,
                    'total_score': 0.0,
                    'scores': [],
                    'success_count': 0
                }

            data_source_stats[data_source]['count'] += 1
            if result['evaluation_status'] == 'success':
                data_source_stats[data_source]['total_score'] += result['main_score']
                data_source_stats[data_source]['scores'].append(
                    result['main_score'])
                data_source_stats[data_source]['success_count'] += 1

        # Calculate averages and additional statistics
        for source, stats in data_source_stats.items():
            if stats['success_count'] > 0:
                stats['average_score'] = stats['total_score'] / \
                    stats['success_count']
                stats['max_score'] = max(stats['scores'])
                stats['min_score'] = min(stats['scores'])
                stats['std_score'] = np.std(stats['scores']) if len(
                    stats['scores']) > 1 else 0.0
            else:
                stats['average_score'] = 0.0
                stats['max_score'] = 0.0
                stats['min_score'] = 0.0
                stats['std_score'] = 0.0

        return {
            'total_samples': total_samples,
            'successful_evaluations': successful_evals,
            'failed_evaluations': failed_evals,
            'success_rate': successful_evals / total_samples if total_samples > 0 else 0.0,
            'data_source_statistics': data_source_stats,
            'overall_average_score': np.mean([r['main_score'] for r in results if r['evaluation_status'] == 'success']) if successful_evals > 0 else 0.0
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any],
                     dataset_name: str):
        """Save evaluation results and statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean model name for filename (remove special characters)
        clean_model_name = self.model_name.replace(
            "/", "_").replace(":", "_").replace("-", "_")
        
        # Add data source suffix if filtering
        data_source_suffix = f"_{'_'.join(self.target_data_sources)}" if self.target_data_sources else ""

        # Save detailed results
        results_file = self.output_dir / \
            f"{clean_model_name}_{dataset_name}{data_source_suffix}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved detailed results to: {results_file}")

        # Save statistics
        stats_file = self.output_dir / \
            f"{clean_model_name}_{dataset_name}{data_source_suffix}_statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)
        logger.info(f"Saved statistics to: {stats_file}")

        # Save summary CSV
        summary_data = []
        for result in results:
            summary_data.append({
                'sample_idx': result['sample_idx'],
                'data_source': result['data_source'],
                'main_score': result['main_score'],
                'evaluation_status': result['evaluation_status'],
                'error': result['error']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / \
            f"{clean_model_name}_{dataset_name}{data_source_suffix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary CSV to: {summary_file}")

    def print_statistics(self, statistics: Dict[str, Any], dataset_name: str):
        """Print evaluation statistics"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {dataset_name.upper()}")
        print(f"{'='*60}")

        print(f"Total samples: {statistics['total_samples']}")
        print(
            f"Successful evaluations: {statistics['successful_evaluations']}")
        print(f"Failed evaluations: {statistics['failed_evaluations']}")
        print(f"Success rate: {statistics['success_rate']:.2%}")
        print(
            f"Overall average score: {statistics['overall_average_score']:.4f}")

        print(f"\nData Source Performance:")
        print(
            f"{'Data Source':<40} {'Count':<8} {'Avg Score':<12} {'Success Rate':<12}")
        print("-" * 72)

        for source, stats in statistics['data_source_statistics'].items():
            success_rate = stats['success_count'] / \
                stats['count'] if stats['count'] > 0 else 0.0
            print(
                f"{source:<40} {stats['count']:<8} {stats['average_score']:<12.4f} {success_rate:<12.2%}")

    def run_evaluation(self, max_samples_per_dataset: Optional[int] = None,
                       num_workers: int = 16, batch_size: int = 100):
        """
        Run complete evaluation on all datasets

        Args:
            max_samples_per_dataset: Maximum samples to evaluate per dataset
            num_workers: Number of parallel workers
        """
        logger.info("Starting VLLM evaluation...")

        all_results = []
        all_statistics = {}

        for dataset_path in self.dataset_paths:
            try:
                # Load dataset
                df = self.load_dataset(dataset_path)
                dataset_name = Path(dataset_path).stem

                # Get data source information
                data_source_counts = self.get_data_source_info(df)

                # Evaluate dataset
                results = self.evaluate_dataset(
                    df, dataset_name,
                    max_samples=max_samples_per_dataset,
                    num_workers=num_workers
                )

                # Calculate statistics
                statistics = self.calculate_statistics(results)

                # Save results
                self.save_results(results, statistics, dataset_name)

                # Print statistics
                self.print_statistics(statistics, dataset_name)

                # Store for final summary
                all_results.extend(results)
                all_statistics[dataset_name] = statistics

            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_path}: {e}")
                continue

        # Generate final summary
        if all_results:
            final_statistics = self.calculate_statistics(all_results)
            combined_name = "combined_all_datasets"
            if self.target_data_sources:
                combined_name = f"combined_all_datasets_{'_'.join(self.target_data_sources)}"
            self.save_results(all_results, final_statistics, combined_name)
            display_name = f"Combined All Datasets - {', '.join(self.target_data_sources)}" if self.target_data_sources else "Combined All Datasets"
            self.print_statistics(final_statistics, display_name)

        logger.info("VLLM evaluation completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="VLLM Model Evaluation for CORL Datasets using Trained Checkpoint")
    parser.add_argument(
        "--dataset1",
        default="./evaluation_data/sampled_test_data_200.parquet",
        help="Path to first parquet dataset (now using cost-optimized 200 samples per data source)"
    )
    parser.add_argument(
        "--dataset2",
        default=None,
        help="Path to second parquet dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per dataset (default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 64)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with limited samples"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name to use for API calls (should match VLLM served model name)"
    )
    parser.add_argument(
        "--api-base",
        default="http://35.220.164.252:3888/v1/",
        help="Base URL for VLLM server API"
    )
    parser.add_argument(
        "--api-key",
        default="sk-Vj1HJkVXtLrRPEVyur5MeW8kfFDTCBGU9pvndDM7DVMNzQnr",
        help="API key for VLLM server (usually EMPTY for local server)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="where2place_point_test",
        help="Specific data source(s) to evaluate. Use comma to separate multiple sources (e.g., 'corl_l3_spatial_understand_val,corl_real_l3_grasping_val')"
    )
    parser.add_argument(
        "--processor-config",
        type=str,
        default='./models/Qwen2.5-VL-7B-Instruct',
        help="Path to processor config or processor name for consistent image processing"
    )

    args = parser.parse_args()

    # Set test mode parameters
    if args.test_mode:
        args.max_samples = 10
        logger.info("Running in test mode with limited samples")

    # Initialize evaluator
    dataset_paths = [args.dataset1]
    if args.dataset2:
        dataset_paths.append(args.dataset2)

    # Parse data sources (comma-separated)
    target_data_sources = None
    if args.data_source:
        target_data_sources = [ds.strip() for ds in args.data_source.split(',')]

    # Initialize processor if provided
    processor = None
    if args.processor_config:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(args.processor_config)
            logger.info(f"Loaded processor: {args.processor_config}")
        except Exception as e:
            logger.warning(f"Failed to load processor {args.processor_config}: {e}")
            logger.info("Continuing without processor (may affect consistency)")

    evaluator = VLLMEvaluator(
        dataset_paths,
        args.output_dir,
        model_name=args.model,
        target_data_sources=target_data_sources,
        api_base=args.api_base,
        api_key=args.api_key,
        processor=processor
    )

    # Run evaluation with optimized settings
    evaluator.run_evaluation(
        max_samples_per_dataset=args.max_samples,
        num_workers=args.workers,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
