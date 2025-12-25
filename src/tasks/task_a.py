TASK_ID = "A"

# Shared schema (all tasks use 'code'; label meaning differs by task)
TEXT_COLUMN = "code"
LABEL_COLUMN = "label"

# Task A: binary labels
# 0 = human, 1 = machine
NUM_LABELS = 2
LABEL_MEANING = {0: "human", 1: "machine"}
