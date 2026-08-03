"""B3 pre-registered configuration (frozen 2026-08-03, 9/10 review locks)."""

# Data
DATA_DIR = '/home/as/vllm/cell/idea-stage/data'
ENDPOINTS = ['hERG', 'AMES', 'BBB_Martins', 'Pgp_Broccatelli',
             'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
             'Bioavailability_Ma']
# 8 binary endpoints (all verified binary; LD50_Zhu EXCLUDED: regression task, Y continuous)

# Splits
N_SPLIT_INSTANCES = 5      # independent split manifests
N_SEEDS = 3                # optimization seeds nested within splits
RANDOM_FRAC = 0.8
GLOBAL_MOLECULE_ALLOCATION = True   # canonical SMILES assigned to one partition across ALL endpoints
MAIN_CLUSTER = 'full_pool_ecfp4'    # locked primary: full unlabeled pool, ECFP4 structural descriptors
SENS_CLUSTER = 'train_anchor'       # sensitivity only
NEW_FINGERPRINT = 'ECFP4'           # primary novelty representation
SENS_FINGERPRINT = 'Morgan2_cosine' # sensitivity only

# Model (frozen hard-sharing MTL treatment)
ARCH = 'MLP'                # hidden (256,128), shared backbone + heads
TASK_SAMPLING = 'equal_expected_contribution'   # primary
SENS_TASK_SAMPLING = 'sample_proportional'
LR = 1e-3
EPOCHS = 60
BATCH = 512
EARLY_STOP = None           # fixed epochs, no data-dependent early stop
CHECKPOINT_SELECTION = 'aggregate_val'          # primary: aggregate validation objective
SENS_CHECKPOINT = 'per_endpoint'
SEED_ESTIMAND = 'average_seed_loss_contrast'    # average seed-level loss differences, NOT probabilities
MAX_TRAIN_PER_ENDPOINT = 8000
MIN_TEST_PER_ENDPOINT = 50
MIN_POS_NEG_PER_BIN = 30    # min class count per novelty bin

# Continuous axis
NOVELTY_METRIC = 'nearest_train_tanimoto'   # relative to TARGET endpoint's training molecules
N_BINS = 8
SPLINE_KNOTS = 4            # prespecified; comparison vs null/linear/monotone/non-monotone

# Inferential pipeline (calibrated by G0 simulation BEFORE main run)
BOOTSTRAP = 'molecule_clustered_paired'     # scaffold-clustered paired bootstrap
CI_LEVEL = 0.90
WINDOW_CRITERION = ['interior_max', 'derivative_increase_then_decrease',
                    'simultaneous_band', 'spline_complexity_stable']
FALSE_WINDOW_RATE_TARGET = 0.05
COVERAGE_TARGET = 0.90

# Development evidence (excluded from confirmatory analysis)
DEV_ENDINGS = ['random', 'scaffold', 'cluster']  # pilot results marked dev-only

SEED_BASE = 20260803
