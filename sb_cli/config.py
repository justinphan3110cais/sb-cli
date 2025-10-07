import os
from enum import Enum

API_BASE_URL = os.getenv("SWEBENCH_API_URL", "https://api.swebench.com")

class Subset(str, Enum):
    swe_bench_m = 'swe-bench-m'
    swe_bench_lite = 'swe-bench_lite'
    swe_bench_verified = 'swe-bench_verified'

# Maximum size for a single prediction submission in MB
MAX_PREDICTION_SIZE_MB = float(os.getenv('SB_CLI_MAX_PREDICTION_SIZE_MB', '10'))
