import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class GroundingDINOEvaluator:
    """Comprehensive evaluation pipeline for Grounding DINO on multiple datasets"""
    
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = None):
        """
        Initialize evaluator with model
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Get model size
        self.model_size_mb = self._get_model_size()
        print(f"Model size: {self.model_size_mb:.2f} MB")
    
    def _get_model_size(self) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def load_dataset(self, ann_file: str, img_prefix: str) -> Tuple[List[Dict], List[str]]:
        """
        Load dataset from instances.json file
        
        Args:
            ann_file: Path to instances.json file with format:
                {
                    "images": [{"id": int, "file_name": str, "width": int, "height": int}],
                    "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h]}],
                    "categories": [{"id": int, "name": str}]
                }
            img_prefix: Path to directory containing images
        
        Returns:
            annotations: List of annotation dictionaries
            class_names: List of class names
        """
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build category mapping
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        class_names = list(set(categories.values()))
        
        # Build image info mapping
        images = {img['id']: img for img in data['images']}
        
        # Group annotations by image
        img_anns = defaultdict(list)
        for ann in data['annotations']:
            img_anns[ann['image_id']].append(ann)
        
        # Build annotation list
        annotations = []
        for img_id, img_info in images.items():
            anns = img_anns[img_id]
            
            boxes = []
            labels = []
            for ann in anns:
                # Handle bbox format - could be [x, y, w, h] (COCO) or [x1, y1, x2, y2]
                bbox = ann['bbox']
                if len(bbox) == 4:
                    # Assume COCO format [x, y, w, h] and convert to [x1, y1, x2, y2]
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                else:
                    boxes.append(bbox)
                
                labels.append(categories[ann['category_id']])
            
            # Construct image path
            file_name = img_info['file_name']
            if os.path.isabs(file_name):
                image_path = file_name
            else:
                image_path = os.path.join(img_prefix, file_name)
            
            annotations.append({
                'image_id': img_id,
                'image_path': image_path,
                'width': img_info['width'],
                'height': img_info['height'],
                'boxes': boxes,
                'labels': labels
            })
        
        return annotations, class_names
    
    def create_few_shot_prompt(self, few_shot_instances: List[Dict], class_names: List[str], 
                                 num_shots: int) -> str:
        """
        Create few-shot text prompt from examples
        
        Args:
            few_shot_instances: List of example instances with 'label' key
            class_names: All possible class names
            num_shots: Number of shots (0, 1, or 5)
        
        Returns:
            Text prompt for the model
        """
        if num_shots == 0:
            # Zero-shot: just list all classes
            return '. '.join(class_names) + '.'
        
        # Few-shot: include example context (simplified - just use class names)
        # In practice, you might want to reference specific examples
        return '. '.join(class_names) + '.'
    
    def predict_image(self, image_path: str, text_prompt: str, 
                      threshold: float = 0.25) -> Dict:
        """
        Run prediction on a single image
        
        Args:
            image_path: Path to image
            text_prompt: Text description of objects to detect
            threshold: Confidence threshold
        
        Returns:
            Dictionary with boxes, scores, and labels
        """
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(image.height, image.width)]
        )
        
        return results[0]
    
    def evaluate_with_timing(self, annotations: List[Dict], text_prompt: str,
                           threshold: float = 0.25) -> Tuple[List, float]:
        """
        Evaluate on dataset and measure inference time
        
        Returns:
            predictions: List of predictions
            avg_inference_time: Average inference time per image in ms
        """
        predictions = []
        inference_times = []
        
        for ann in annotations:
            image_path = ann['image_path']
            
            # Time the prediction
            start_time = time.time()
            pred = self.predict_image(image_path, text_prompt, threshold)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            predictions.append({
                'image_id': ann['image_id'],
                'boxes': pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes'],
                'scores': pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores'],
                'labels': pred['labels']
            })
        
        avg_inference_time = np.mean(inference_times)
        return predictions, avg_inference_time
    
    def compute_metrics_coco_format(self, predictions: List[Dict], 
                                     ground_truth: List[Dict],
                                     class_names: List[str]) -> Dict:
        """
        Compute mAP metrics in COCO format
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            class_names: List of class names
        
        Returns:
            Dictionary with mAP, mAP50, mAP75
        """
        # Convert to COCO format
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        # Build COCO ground truth
        coco_gt = {
            'images': [],
            'annotations': [],
            'categories': [{'id': idx, 'name': name} for idx, name in enumerate(class_names)]
        }
        
        ann_id = 0
        for gt in ground_truth:
            img_id = gt['image_id'] if isinstance(gt['image_id'], int) else hash(str(gt['image_id'])) % (10**8)
            
            coco_gt['images'].append({
                'id': img_id,
                'width': gt['width'],
                'height': gt['height']
            })
            
            for box, label in zip(gt['boxes'], gt['labels']):
                x1, y1, x2, y2 = box
                coco_gt['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': class_to_id.get(label, 0),
                    'bbox': [x1, y1, x2-x1, y2-y1],  # COCO format: [x, y, width, height]
                    'area': (x2-x1) * (y2-y1),
                    'iscrowd': 0
                })
                ann_id += 1
        
        # Build COCO predictions
        coco_pred = []
        for pred in predictions:
            img_id = pred['image_id'] if isinstance(pred['image_id'], int) else hash(str(pred['image_id'])) % (10**8)
            
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                coco_pred.append({
                    'image_id': img_id,
                    'category_id': class_to_id.get(label, 0),
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
        
        # Evaluate using COCO API
        if len(coco_pred) == 0:
            return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}
        
        # Create temporary files for COCO evaluation
        with open('/tmp/coco_gt.json', 'w') as f:
            json.dump(coco_gt, f)
        
        coco_gt_api = COCO('/tmp/coco_gt.json')
        coco_dt = coco_gt_api.loadRes(coco_pred)
        
        coco_eval = COCOeval(coco_gt_api, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'mAP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
            'mAP50': coco_eval.stats[1],    # AP @ IoU=0.50
            'mAP75': coco_eval.stats[2],    # AP @ IoU=0.75
        }
        
        return metrics
    
    def evaluate_dataset(self, dataset_name: str, ann_file: str, img_prefix: str,
                         few_shot_file: str = None, num_shots: int = 0) -> Dict:
        """
        Evaluate on a specific dataset with few-shot configuration
        
        Args:
            dataset_name: Name of dataset (for logging purposes)
            ann_file: Path to instances.json annotation file
            img_prefix: Path to directory containing images
            few_shot_file: Path to few-shot instances JSON file
            num_shots: Number of shots (0, 1, or 5)
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name.upper()} - {num_shots}-shot")
        print(f"{'='*60}")
        
        # Load dataset from instances.json
        annotations, class_names = self.load_dataset(ann_file, img_prefix)
        print(f"Loaded {len(annotations)} images with {len(class_names)} classes")
        
        # Load few-shot instances if provided
        few_shot_instances = []
        if few_shot_file and num_shots > 0:
            with open(few_shot_file, 'r') as f:
                few_shot_data = json.load(f)
                few_shot_instances = few_shot_data.get('instances', [])[:num_shots]
        
        # Create text prompt
        text_prompt = self.create_few_shot_prompt(few_shot_instances, class_names, num_shots)
        print(f"Text prompt: {text_prompt[:100]}...")
        
        # Run evaluation
        print(f"Running inference on {len(annotations)} images...")
        predictions, avg_inference_time = self.evaluate_with_timing(
            annotations, text_prompt, threshold=0.25
        )
        
        # Compute metrics
        print("Computing metrics...")
        metrics = self.compute_metrics_coco_format(predictions, annotations, class_names)
        
        # Add additional info
        metrics['inference_time_ms'] = avg_inference_time
        metrics['model_size_mb'] = self.model_size_mb
        metrics['num_shots'] = num_shots
        metrics['dataset'] = dataset_name
        
        return metrics
    
    def run_full_evaluation(self, config: Dict) -> Dict:
        """
        Run complete evaluation across all datasets and shot configurations
        
        Args:
            config: Configuration dictionary with:
                - datasets: Dict with dataset configurations, each containing:
                    - ann_file: Path to instances.json
                    - img_prefix: Path to image directory
                - few_shot_files: Dict with paths to few-shot files for each dataset
                - shots: List of shot numbers to evaluate [0, 1, 5]
        
        Returns:
            Complete results dictionary
        """
        all_results = {}
        
        for dataset_name, dataset_config in config['datasets'].items():
            all_results[dataset_name] = {}
            
            for num_shots in config['shots']:
                few_shot_file = config.get('few_shot_files', {}).get(dataset_name)
                
                try:
                    metrics = self.evaluate_dataset(
                        dataset_name=dataset_name,
                        ann_file=dataset_config['ann_file'],
                        img_prefix=dataset_config['img_prefix'],
                        few_shot_file=few_shot_file,
                        num_shots=num_shots
                    )
                    all_results[dataset_name][f'{num_shots}shot'] = metrics
                    
                    # Print results
                    print(f"\nResults for {dataset_name} {num_shots}-shot:")
                    print(f"  mAP:              {metrics['mAP']:.4f}")
                    print(f"  mAP@0.5:          {metrics['mAP50']:.4f}")
                    print(f"  mAP@0.75:         {metrics['mAP75']:.4f}")
                    print(f"  Inference time:   {metrics['inference_time_ms']:.2f} ms/image")
                    print(f"  Model size:       {metrics['model_size_mb']:.2f} MB")
                    
                except Exception as e:
                    print(f"Error evaluating {dataset_name} {num_shots}-shot: {e}")
                    all_results[dataset_name][f'{num_shots}shot'] = {'error': str(e)}
        
        return all_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation script"""
    
    # Configuration - uses ann_file and img_prefix for each dataset
    config = {
        'datasets': {
            'coco': {
                'ann_file': '/path/to/coco/annotations/instances_val2017.json',  # Update this path
                'img_prefix': '/path/to/coco/val2017',  # Update this path
            },
            'voc': {
                'ann_file': '/path/to/voc/instances.json',  # Update this path
                'img_prefix': '/path/to/voc/JPEGImages',  # Update this path
            },
            'beetle': {
                'ann_file': '/path/to/beetle/instances.json',  # Update this path
                'img_prefix': '/path/to/beetle/images',  # Update this path
            }
        },
        'few_shot_files': {
            'coco': '/path/to/coco_few_shot.json',  # Update this path
            'voc': '/path/to/voc_few_shot.json',    # Update this path
            'beetle': '/path/to/beetle_few_shot.json',  # Update this path
        },
        'shots': [0, 1, 5],
        'output_file': 'grounding_dino_evaluation_results.json'
    }
    
    # Initialize evaluator
    evaluator = GroundingDINOEvaluator(
        model_id="IDEA-Research/grounding-dino-tiny",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation(config)
    
    # Save results
    evaluator.save_results(results, config['output_file'])
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'Shots':<8} {'mAP':<10} {'mAP@0.5':<10} {'mAP@0.75':<10} {'Time(ms)':<12} {'Size(MB)':<10}")
    print("-"*80)
    
    for dataset in results:
        for shot_config in results[dataset]:
            if 'error' not in results[dataset][shot_config]:
                m = results[dataset][shot_config]
                shot_num = shot_config.replace('shot', '')
                print(f"{dataset:<15} {shot_num:<8} {m['mAP']:<10.4f} {m['mAP50']:<10.4f} "
                      f"{m['mAP75']:<10.4f} {m['inference_time_ms']:<12.2f} {m['model_size_mb']:<10.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
