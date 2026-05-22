# Experiment Instructions

The code for running inference, evaluating inference results, and training LoRA models is inside the `scripts/` folder.

The YAML config files are inside the `configs/` folder.

## General Workflow

To run a new experiment, you need to create or update a YAML config file.

The easiest way to do this is usually:

1. Duplicate an existing YAML config file.
2. Rename it for the new experiment.
3. Change the parameters you want to test.
4. Update the paths so they match the new experiment.

This makes experiments easier to reproduce and compare.

Instead of going into the actual Python code and changing parameters there, each experiment should have its own YAML config file. That way, we can keep track of exactly which settings were used for each run.

---

## Running Experiments

After creating the YAML config file, run the matching command depending on what you want to do.

---

## Train a LoRA Model

Use this command to train a LoRA model:

```bash
python scripts/train_lora.py --config configs/YOUR_CONFIG_FILE.yaml 2>&1 | tee outputs/YOUR_LOG_FILE.log
```

---

## Generate Responses

Use this command to run inference and generate model responses:

```bash
python scripts/generate_responses.py --config configs/YOUR_CONFIG_FILE.yaml 2>&1 | tee outputs/YOUR_LOG_FILE.log
```

---

## Evaluate Responses

Use this command to evaluate generated responses:

```bash
python scripts/evaluate_responses.py --config configs/YOUR_CONFIG_FILE.yaml
```

---

## Logs

For training and response generation, the terminal output is also saved to a log file.

This part of the command saves the output:

```bash
2>&1 | tee outputs/YOUR_LOG_FILE.log
```

This is useful because you can check the log file later to see what happened during the run.

The log files are saved inside the `outputs/` folder.

---

## Useful Commands

### Activate the Virtual Environment

Before running the scripts, activate the virtual environment:

```bash
source /scratch/.venv/bin/activate
```

---

## Using `tmux`

`tmux` lets you create a terminal session that keeps running even if you close your laptop or lose connection.

This is useful for long training or inference jobs.

### Create a New `tmux` Session

```bash
tmux new -s train
```

This creates a new session named `train`.

You can replace `train` with any session name you want.

### Detach from a `tmux` Session

To leave the `tmux` session without stopping the process, press:

```text
Ctrl + b
```

Then press:

```text
d
```

This detaches you from the session, but the command will keep running in the background.

### Reattach to an Existing `tmux` Session

```bash
tmux attach -t train
```

This reconnects you to the existing `train` session.

---

