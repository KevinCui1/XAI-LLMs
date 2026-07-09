"""
Settings
"""

import os

CUDA = False
ALPHA = None  # Use "None" to use ReLU threshold (i.e., > 0)
BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 5  # original research value
COMPLEXITY_PENALTY = 1.00
TOPN = 5
DEBUG = False

# Choices: iou, precision, recall
METRIC = "iou"

EMBEDDING_NEIGHBORHOOD_SIZE = 5

NEURONS = list(range(30))
# hardcoded: os.cpu_count() reports the host's total cores (256), not the container's CPU limit.
# Sized to memory, not CPUs: a single unit's beam search peaks ~16.4GiB, so 4 workers
# (4 x ~13GiB private + ~4GiB shared base) stays under the 64Gi pod limit. Per-worker torch
# intra-op threads are pinned to 1 in analyze._init_worker so 4 workers ~ 4 of the 8 CPUs.
PARALLEL = 4

SHUFFLE = False
SAVE_EVERY = 4

# How many "maximally activating" open features to use, PER CATEGORY
MAX_OPEN_FEATS = 5
# Minimum number of activations to analyze a neuron
MIN_ACTS = 500

MODEL = "models/bowman_snli/6.pth"
MODEL_TYPE = "bowman"  # choices: bowman, minimal
RANDOM_WEIGHTS = False  # Initialize weights randomly (equivalent to an untrained model)
N_SENTENCE_FEATS = 2000  # original research value

DATA = "data/analysis/snli_1.0_dev.feats"

assert DATA.endswith(".feats")
VECPATH = DATA.replace(".feats", ".vec")

# Overridables
if "MTDISSECT_MODEL" in os.environ:
    MODEL = os.environ["MTDISSECT_MODEL"]
if "MTDISSECT_MAX_FORMULA_LENGTH" in os.environ:
    MAX_FORMULA_LENGTH = int(os.environ["MTDISSECT_MAX_FORMULA_LENGTH"])
if "MTDISSECT_MAX_OPEN_FEATS" in os.environ:
    MAX_OPEN_FEATS = int(os.environ["MTDISSECT_MAX_OPEN_FEATS"])
if "MTDISSECT_METRIC" in os.environ:
    METRIC = os.environ["MTDISSECT_METRIC"]

mbase = os.path.splitext(os.path.basename(MODEL))[0]
dbase = os.path.splitext(os.path.basename(DATA))[0]
RESULT = f"exp/{dbase}-{mbase}-sentence-{MAX_FORMULA_LENGTH}{'-shuffled' if SHUFFLE else ''}{'-debug' if DEBUG else ''}{f'-{METRIC}' if METRIC != 'iou' else ''}{f'-random-weights' if RANDOM_WEIGHTS else ''}"

print(RESULT)
