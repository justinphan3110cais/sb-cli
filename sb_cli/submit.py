import json
import time
import requests
import typer
import sys
from typing import Optional
from typing_extensions import Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
from sb_cli.config import API_BASE_URL, Subset, MAX_PREDICTION_SIZE_MB
from sb_cli.get_report import get_report
from sb_cli.utils import verify_response
from pathlib import Path

app = typer.Typer(help="Submit predictions to the SBM API")

def submit_prediction(prediction: dict, headers: dict, payload_base: dict):
    """Submit a single prediction."""
    payload = payload_base.copy()
    payload["prediction"] = prediction
    response = requests.post(f'{API_BASE_URL}/submit', json=payload, headers=headers)
    verify_response(response)
    return response.json()


def check_prediction_size(prediction: dict, max_size_mb: float = MAX_PREDICTION_SIZE_MB) -> tuple[bool, float]:
    """
    Check if a prediction exceeds the size limit.
    Returns (is_within_limit, size_in_mb).
    """
    json_str = json.dumps(prediction)
    size_bytes = len(json_str.encode('utf-8'))
    size_mb = size_bytes / (1024 * 1024)
    return (size_mb <= max_size_mb, size_mb)

# Prediction Processing
def process_predictions(predictions_path: str, instance_ids: list[str]):
    """Load and validate predictions from file."""
    with open(predictions_path, 'r') as f:
        if predictions_path.endswith('.json'):
            predictions = json.load(f)
        else:
            predictions = [json.loads(line) for line in f]
    preds = []
    if isinstance(predictions, list):
        for p in predictions:
            instance_id = p['instance_id']
            if instance_ids and instance_id not in instance_ids:
                continue
            preds.append({
                'instance_id': instance_id,
                'model_patch': p['model_patch'],
                'model_name_or_path': p['model_name_or_path']
            })
    else:
        for instance_id, p in predictions.items():
            if instance_ids and instance_id not in instance_ids:
                continue
            preds.append({
                'instance_id': instance_id,
                'model_patch': p['model_patch'],
                'model_name_or_path': p['model_name_or_path']
            })
    if len(set([p['model_name_or_path'] for p in preds])) > 1:
        raise ValueError("All predictions must be for the same model")
    if len(set([p['instance_id'] for p in preds])) != len(preds):
        raise ValueError("Duplicate instance IDs found in predictions - please remove duplicates before submitting")
    # Check prediction sizes and warn about large ones
    size_warnings = []
    valid_preds = []
    for pred in preds:
        is_valid, size_mb = check_prediction_size(pred)
        if not is_valid:
            size_warnings.append({
                'instance_id': pred['instance_id'],
                'size_mb': size_mb,
                'patch_length': len(pred.get('model_patch', ''))
            })
        else:
            valid_preds.append(pred)
    
    if size_warnings:
        console = Console()
        console.print(f"[yellow]Warning: {len(size_warnings)} predictions exceed {MAX_PREDICTION_SIZE_MB}MB size limit and will be skipped:[/]")
        for warning in size_warnings[:10]:  # Show first 10
            console.print(f"  - {warning['instance_id']}: {warning['size_mb']:.2f}MB (patch: {warning['patch_length']:,} chars)")
        if len(size_warnings) > 10:
            console.print(f"  ... and {len(size_warnings) - 10} more")
        console.print(f"[yellow]  Set SB_CLI_MAX_PREDICTION_SIZE_MB environment variable to increase limit[/]")
    
    return valid_preds
