from datasets import load_dataset

DATASET_NAME = "DaniilOr/SemEval-2026-Task13"

def load_task(task_id: str, train_size=None, val_size=None, test_size=None):
    """
    Load a SemEval-2026 Task13 subtask by id: 'A', 'B', or 'C'.
    Optionally subsample splits for faster experiments.
    """
    ds = load_dataset(DATASET_NAME, task_id)

    train_ds = ds["train"]
    val_ds = ds.get("validation")
    test_ds = ds.get("test")

    if train_size is not None:
        train_ds = train_ds.select(range(train_size))
    if val_ds is not None and val_size is not None:
        val_ds = val_ds.select(range(val_size))
    if test_ds is not None and test_size is not None:
        test_ds = test_ds.select(range(test_size))

    return train_ds, val_ds, test_ds
