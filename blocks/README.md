blocks — Project layout (reorganized)

Overview

This folder provides a clear, navigable grouping of the repository by purpose. It contains three child folders:

- `main`: Core system code (the files that implement the application and retrieval logic).
- `experiments`: Experiment code, configs and orchestration scripts.
- `result`: Experiment outputs and result artifacts.

Directories

- `blocks/main/`
  - Contains the application logic and utilities (moved here from repository root). Example files:
    - `main.py`, `chatbot.py`, `comparison_chatbot.py`, `retrieval_strategies.py`, `add_embeddings.py`, `create_test_dataset.py`, `inspect_db.py`, `check_requirements.py`

- `blocks/experiments/`
  - Experiment drivers, analysis and configs. Example files:
    - `run_experiments.py`, `experimental_design.py`, `evaluation_framework.py`, `analyze_comparisons.py`, `statistical_analysis.py`, `test_threshold_tuning.py`, `run_research_workflow.sh`, `experiment_config.json`, `test_queries.json`

- `blocks/result/`
  - All generated or dataset/result artifacts. Example files/directories:
    - `experiment_results_intermediate.json`, `experiment_results_raw.json`, `experiment_metrics.csv`, `experiment_test_queries.json`, `threshold_tuning_results.json`, `comparison_logs.jsonl`, `statistical_analysis_results/`

How to run

From the repository root, run the scripts directly. Examples:

- Run the main application (CLI):

  python blocks/main/main.py

  or

  python -m blocks.main.main

- Run experiments:

  python blocks/experiments/run_experiments.py

Where outputs appear

- Experiment outputs and intermediate files are stored in `blocks/result/`.

Notes & next steps

- I moved files (not copied). If tooling or imports expect the old layout you may need to update references or add the repository root to `PYTHONPATH` when running from other locations. Running the scripts from the project root should work without changes.
- If you want, I can update top-level entrypoints, install small import wrappers, or update README with recommended `venv` commands.
