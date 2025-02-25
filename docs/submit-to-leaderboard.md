# Submit to Multimodal Leaderboard

This guide explains how to submit your system's results to the [SWE-bench Multimodal](https://www.swebench.com/multimodal) leaderboard.

## Prerequisites

Before submitting, ensure you have:

- Completed the [Quick Start](quick-start.md) guide
- Generated predictions for the `swe-bench-m` / `test` split

## Submission Steps

1. **Submit Predictions to API**

First, submit your predictions to the SWE-bench API:

```bash
sb-cli submit swe-bench-m test \
    --predictions_path ./path/to/preds.json \
    --run_id your-run-id
```

2. **Prepare GitHub Submission**

Fork and clone the [experiments repository](https://github.com/swe-bench/experiments):

```bash
git clone https://github.com/YOUR_USERNAME/experiments.git
cd experiments
```

3. **Create Submission Files**

Create a new directory for your submission and add the following files:

```
experiments/
└── multimodal/
    └── YOUR_MODEL_NAME/
        ├── README.md         # Description of your model/approach
        └── metadata.yaml     # Submission metadata
```

Example `metadata.yaml`:
```yaml
name: "Your System's Name"
oss: true  # Whether your system is open source
site: "https://github.com/..."  # URL to link to from the leaderboard
```

4. **Submit Pull Request**

Create a pull request to the [experiments repository](https://github.com/swe-bench/experiments) with your submission files.

In addition the files above, you should include the following details in the PR's description:
1. Email used for sb-cli submission
2. The `run_id` used for sb-cli submission

The SWE-bench team will:
1. Add your predictions and results to the PR
2. Review and merge your submission
3. Update the leaderboard with your results

## Notes

- Make sure your `run_id` matches between the API submission and metadata file
- We maintain this leaderboard for free, we try to update it as soon as possible after submission
- Test split submissions are limited - please check your quota using `sb-cli quota swe-bench-m test` before submitting so you don't run out. Quotas are reloaded every 30 days.

For questions about the submission process, please open an issue in the [experiments repository](https://github.com/swe-bench/experiments/issues).
