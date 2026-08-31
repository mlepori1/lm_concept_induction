import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Procrustes rotation
# ──────────────────────────────────────────────────────────────────────────────

def fit_orthogonal_procrustes(P_train, D_train):
    """Find the rotation matrix R that best maps prototype vectors to displacement vectors.

    We want P_train @ R ≈ D_train, minimising ||D_train − P_train @ R||_F subject to R
    being orthogonal with det = +1.

    Closed-form solution: take the SVD of the cross-covariance P^T D = U S V^T,
    then R = U @ V^T

    P is centered internally: since R must be orthogonal it cannot rescale or shift vectors,
    so if P has a large mean but D is zero-mean, it messses up the optimization

    Args:
        P_train: (n, d) training prototypes
        D_train: (n, d) training displacements (mean-centered)

    Returns:
        R:      (d, d) rotation matrix
        P_mean: (d,)   prototype mean subtracted before fitting (needed at predict time)
    """
    P_mean  = P_train.mean(axis=0)
    P_train = P_train - P_mean

    U, _, Vt = np.linalg.svd(P_train.T @ D_train)  # (d, d) cross-covariance

    # Ensure proper rotation (det = +1), not a reflection (det = -1)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    return U @ Vt, P_mean


def predict_procrustes(P_new, R, P_mean):
    """Apply the Procrustes rotation to new prototypes.

    Centers by P_mean (from fit) before rotating. Returns (P_new - P_mean) @ R, shape (m, d).
    """
    return (P_new - P_mean) @ R

# ──────────────────────────────────────────────────────────────────────────────
# MLP regression
# ──────────────────────────────────────────────────────────────────────────────

class MLP(torch.nn.Module):
    """Two-layer MLP (Linear -> GELU -> Linear) mapping a prototype to a displacement.

    More expressive comparison point than the global ridge/orthogonal maps above,
    while being easier to optimize than a per-example rank-one hypernetwork.
    """

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, prototype):
        return self.net(prototype)


def _train_mlp_epochs(P_fit_t, D_fit_t, P_val_t, D_val_t, hidden_dim, d,
                       device, max_epochs, lr, weight_decay, patience, seed):
    """Train one MLP for up to max_epochs full-batch steps, recording val cosine
    similarity every epoch and early-stopping after `patience` epochs without an
    improvement. The gradient-descent loss is still MSE (needed for a differentiable
    objective), but we track mean cosine similarity -- the metric we actually report --
    directly instead of using MSE as a proxy.

    Returns:
        val_cos: (max_epochs,) array, val_cos[e] = mean cosine similarity between
            predicted and true displacement on (P_val_t, D_val_t) after e+1 gradient
            steps at this weight_decay. Once early-stopped, remaining entries are
            padded with the best value seen (a flat continuation), so the array is
            always length max_epochs and safe to aggregate across folds/candidates
            that stop at different epochs.
    """
    torch.manual_seed(seed)
    model = MLP(hidden_dim, d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    D_val = D_val_t.cpu().numpy()
    val_cos = np.empty(max_epochs)
    best_so_far = -np.inf
    epochs_since_improvement = 0
    stopped_epoch = max_epochs - 1

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(P_fit_t), D_fit_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(P_val_t).cpu().numpy()
        cur_cos = batch_cosine(pred_val, D_val).mean()
        val_cos[epoch] = cur_cos

        if cur_cos > best_so_far:
            best_so_far = cur_cos
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            stopped_epoch = epoch
            break

    val_cos[stopped_epoch + 1:] = best_so_far
    return val_cos


def fit_mlp(P_train, D_train, hidden_dim, device, max_epochs, lr, patience, wd_grid, cv_folds, seed):
    """Train an MLP to map prototypes to displacements.

    Selects weight_decay and number of training epochs via cv_folds cross-validation
    over (P_train, D_train), maximizing mean cosine similarity between predicted and true
    displacement on the held-out fold. Ridge is scored the exact same way (see COSINE_SCORER)
    with the exact same cv_folds,  so the comparison to ridge is apples-to-apples. 
    Each (fold, weight_decay) candidate is early-stopped after `patience` epochs without 
    improvement (see _train_mlp_epochs), so this doesn't always cost max_epochs steps. 
    The winning (weight_decay, n_epochs) pair is then used to retrain a model from scratch on the 
    full training set, mirroring how RidgeCV refits on all the data at its chosen alpha.

    Args:
        P_train, D_train: (n, d) prototypes/displacements
        hidden_dim: width of the MLP's hidden layer
        device: torch device to train on
        max_epochs: max full-batch gradient steps to try per candidate/fold
        lr: Adam learning rate
        patience: early-stop a (fold, weight_decay) candidate after this many epochs
            without a val cosine-similarity improvement
        wd_grid: candidate Adam weight_decay values to cross-validate
        cv_folds: number of CV folds (mirrors RidgeCV's cv argument)
        seed: random seed, shared across the KFold split, weight init, and final refit

    Returns:
        model:       final MLP retrained on the full training set, in eval mode
        best_wd:     weight_decay value selected by CV
        best_epochs: number of training epochs selected by CV
        cv_cos:      mean held-out-fold cosine similarity at (best_wd, best_epochs)
        fit_cos:     full-training-set mean cosine similarity of the final refit model
    """
    d = P_train.shape[1]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # cv_cos[wd_idx, epoch] = mean cosine similarity at that epoch, averaged over folds
    cv_cos = np.zeros((len(wd_grid), max_epochs))
    for train_idx, val_idx in kf.split(P_train):
        P_fit_t = torch.tensor(P_train[train_idx], dtype=torch.float32, device=device)
        D_fit_t = torch.tensor(D_train[train_idx], dtype=torch.float32, device=device)
        P_val_t = torch.tensor(P_train[val_idx], dtype=torch.float32, device=device)
        D_val_t = torch.tensor(D_train[val_idx], dtype=torch.float32, device=device)

        # Fit an MLP for each weight_decay in the grid
        for wd_idx, wd in enumerate(wd_grid):
            cv_cos[wd_idx] += _train_mlp_epochs(
                P_fit_t, D_fit_t, P_val_t, D_val_t, hidden_dim, d,
                device, max_epochs, lr, wd, patience, seed,
            )
    # Average over folds
    cv_cos /= cv_folds

    best_wd_idx, best_epoch_idx = np.unravel_index(np.argmax(cv_cos), cv_cos.shape)
    best_wd = wd_grid[best_wd_idx]
    best_epochs = int(best_epoch_idx) + 1
    cv_score = float(cv_cos[best_wd_idx, best_epoch_idx])

    # Refit on the full training set at (best_wd, best_epochs)
    P_train_t = torch.tensor(P_train, dtype=torch.float32, device=device)
    D_train_t = torch.tensor(D_train, dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    model = MLP(hidden_dim, d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=best_wd)
    criterion = torch.nn.MSELoss()
    # Train the model for the best number of epochs, with the best weight_decay
    for _ in range(best_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(P_train_t), D_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the full training set
    model.eval()
    with torch.no_grad():
        fit_pred = model(P_train_t).cpu().numpy()
    fit_cos = float(batch_cosine(fit_pred, D_train).mean())

    return model, best_wd, best_epochs, cv_score, fit_cos


def predict_mlp(model, P_new, device):
    """Apply a trained MLP to new prototypes.

    Returns predicted displacements, shape (m, d), as a numpy array.
    """
    with torch.no_grad():
        pred = model(torch.tensor(P_new, dtype=torch.float32, device=device))
    return pred.cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Ridge regression hparams
# ──────────────────────────────────────────────────────────────────────────────

ALPHA_GRID = [10.0 ** k for k in range(6)]   # 1, 10, 100, ..., 1e5

MLP_WD_GRID = [0.0, 1e-4, 1e-3, 1e-2]        # candidate Adam weight_decay values (L2, cf. ALPHA_GRID)


# ──────────────────────────────────────────────────────────────────────────────
# Ridge map SVD (rank analysis)
# ──────────────────────────────────────────────────────────────────────────────

def ridge_svd(model, k):
    """Compute the top-k LEFT singular vectors of the ridge map A = coef_.T.

    The map is applied as D = P @ A 
    SVD of A = U S Vt:
      - Left  SVs (columns of U) = right SVs of A^T = INPUT  directions (prototype space)
      - Right SVs (rows   of Vt) = left  SVs of A^T = OUTPUT directions (displacement space)
    We return the left SVs so that cosine similarity with token embeddings tells us
    which prototype directions receive the largest anisotropic gain.

    Args:
        model: fitted RidgeCV model
        k:     int, number of top singular vectors to return

    Returns:
        sigma:  (d,) all singular values in descending order (A is always a square (d, d) map)
        U_top:  (k, d) top-k left singular vectors (input directions of the map)
    """
    A = model.coef_.T   # RidgeCV stores coef_ as (n_targets, n_features); transpose
    U, sigma, _ = np.linalg.svd(A, full_matrices=False)
    return sigma, U.T[:k]


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def batch_cosine(A, B):
    """Row-wise cosine similarity between two (m, d) arrays.

    Returns an (m,) array where entry i = cos(A[i], B[i]).
    """
    A_unit = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-8)
    B_unit = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-8)
    return (A_unit * B_unit).sum(axis=1)


def mean_cosine_similarity(y_true, y_pred):
    """Mean row-wise cosine similarity; the metric function backing COSINE_SCORER."""
    return float(batch_cosine(y_pred, y_true).mean())


# Scores RidgeCV's alpha candidates by mean held-out-fold cosine similarity rather than
# its default, so ridge and the MLP select their hyperparameters by the same criterion
# we actually report: cosine similarity.
COSINE_SCORER = make_scorer(mean_cosine_similarity, greater_is_better=True)


if __name__ == "__main__":
    """Procrustes vs. ridge vs. MLP prototype-to-displacement mapping analysis.

    This script measures:
        1) Whether embedding prototype vectors are predictive of context displacement vectors
        2) Whether this mapping is just a rotation, requires anisotropic scaling, 
            (i.e., whether the model has learned to preferentially accentuate some features)
            or requires nonlinear mapping

    We fit a rotation matrix vs. a L2 penalized regression to perform this mapping,
    and see how performance varies over layers. We also fit a MLP to perform this mapping.

    Output files (under {save_root}/{model_name_safe}/displacement_regression/:
    regressions.csv          per (category, support_id, layer): cosine similarities
    hyperparameter_records.csv  CV-selected ridge alpha + MLP weight_decay/epochs per layer
    singular_values.npy      (n_layers, r) singular values of A^ℓ
    top_input_svecs.npy      (n_layers, k, d) top-k right singular vectors of A^ℓ
    hyperparams.json
    best_A.npy               (d, d) full ridge coefficient matrix at the best layer
    best_layer.json          which layer was picked, and by what metric

    With --leave_one_concept_out, additionally (at the best layer only):
    loco_regressions.csv     per (category, support_id): cosine similarity of a ridge map,
                              a Procrustes rotation, and an MLP, each trained on all OTHER
                              concepts, evaluated on this concept's held-out rows -- plus the
                              (map-independent) shuffled-order noise ceiling for that row
    loco_A.npy                (n_concepts, d, d) one ridge coefficient matrix per concept,
                              each fit excluding that concept's train rows
    loco_categories.json     concept names, in the order matching loco_A.npy's first axis

    With --save_all_A, additionally saves all_A.npy (n_layers, d, d): every layer's ridge
    map, not just the best layer's.

    With --external_A_path, additionally adds cos_external_A/cos_external_A_shuffled
    columns to regressions.csv: at each layer, the held-out cosine similarity of the
    prediction from the loaded external ridge map (from all_A.npy of a prior, typically
    different-dataset run), instead of a map fit on this run's own training data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="HF model id")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="K-fold CV folds for selecting ridge alpha and MLP weight_decay/epochs per layer")
    parser.add_argument("--n_svd_components", type=int, default=100,
                        help="Top-k right singular vectors of A to save for later inspection")
    parser.add_argument("--mlp_hidden_dim", type=int, default=2304,
                        help="Hidden width of the MLP comparison point")
    parser.add_argument("--mlp_max_epochs", type=int, default=2000,
                        help="Max full-batch training epochs to consider when CV-selecting epoch count for the MLP")
    parser.add_argument("--mlp_patience", type=int, default=50,
                        help="Early-stop an MLP CV candidate after this many epochs without a val cosine-similarity improvement")
    parser.add_argument("--mlp_lr", type=float, default=1e-3,
                        help="Adam learning rate for the MLP")
    parser.add_argument("--save_root", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--leave_one_concept_out", action="store_true", default=False,
                        help="After picking the best layer, additionally fit one ridge map, "
                             "Procrustes rotation, and MLP per concept -- excluding that concept's "
                             "train rows -- and evaluate them on that concept's held-out rows, to "
                             "check how well each map generalizes to a wholly unseen concept. "
                             "Saves loco_A.npy for behavioral_modeling.py's own "
                             "--leave_one_concept_out.")
    parser.add_argument("--save_all_A", action="store_true", default=False,
                        help="Save every layer's ridge map to all_A.npy (n_layers, d, d), not "
                             "just the best layer's best_A.npy, so it can be loaded elsewhere "
                             "via --external_A_path for cross-dataset comparisons.")
    parser.add_argument("--external_A_path", type=str, default=None,
                        help="Path to an all_A.npy from a prior run (see --save_all_A), typically "
                             "on a different dataset. If given, at every layer this run's held-out "
                             "prototypes are ALSO predicted with that run's ridge map instead of "
                             "one fit here.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -- this script requires a GPU.")
    device = torch.device("cuda")

    # Shared splitter for every RidgeCV fit below (main loop + LOCO), so ridge sees the
    # exact same fold partition as fit_mlp's own KFold(shuffle=True, random_state=seed).
    ridge_cv_splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # ── Load harvest_model_data.py's output ─────────────────────────────────────
    model_name_safe = args.model_name.replace("/", "_")
    data_dir = os.path.join(args.save_root, model_name_safe, "concept_list_data")

    records = pd.read_csv(
        os.path.join(data_dir, "records.csv"),
        keep_default_na=False, # If a word is "nan" in a dataset, it should be treated as a string, not a float.
        na_values={"log_p_target_given_shuffled_context": [""]}, # Keep NaNs for the log_p_target_given_shuffled_context column.
    )
    prototypes   = np.load(os.path.join(data_dir, "prototypes.npy"))               # (n_records, d)
    displacements = np.load(os.path.join(data_dir, "displacements.npy"))           # (n_records, n_layers, d)

    # Shuffled displacements are used to compute the noise ceiling.
    shuffled_displacements = np.load(os.path.join(data_dir, "shuffled_displacements.npy"))  # (n_heldout, n_layers, d)

    n_layers = displacements.shape[1]

    external_A = None
    if args.external_A_path:
        external_A = np.load(args.external_A_path)   # (n_layers_ext, d, d)
        assert external_A.shape[0] == n_layers, (
            f"--external_A_path has {external_A.shape[0]} layers, this dataset has {n_layers}; "
            f"they must come from the same model"
        )
        print(f"Loaded external A matrices from {args.external_A_path}")

    train_record_indices   = records.index[records["split"] == "train"].tolist()
    heldout_record_indices = records.index[records["split"] == "heldout"].tolist()

    P_train   = prototypes[train_record_indices]     # (n_train, d)
    P_heldout = prototypes[heldout_record_indices]    # (n_heldout, d)

    rng = np.random.default_rng(args.seed)

    # Establish output lists
    output_rows   = []    # one row per (held-out support set, layer)
    hyperparameter_records = []    # best ridge alpha + MLP wd/epochs per layer
    all_sigmas    = []    # singular values of A per layer
    all_top_svecs = []    # top-k input singular vectors of A per layer
    mean_cos_ridge_by_layer = []   # held-out ridge cosine per layer, for reporting only
    ridge_cv_cos_by_layer = []     # train-only k-fold CV ridge cosine per layer, for picking the best layer for behavioral_modeling.py
    all_ridge_models = []           # ridge model per layer, so we can save the best one's A matrix
    mean_cos_external_by_layer = []  # held-out external-A cosine per layer, for reporting only (if external_A_path is provided)

    for layer_idx in tqdm(range(n_layers), desc="Fitting maps per layer"):
        # Displacement matrices for this layer (train and held-out)
        D_train            = displacements[train_record_indices, layer_idx]     # (n_train, d)
        D_heldout          = displacements[heldout_record_indices, layer_idx]   # (n_heldout, d)

        # For the noise ceiling, we shuffle the support and get the resulting context displacement vectors.
        D_heldout_shuffled = shuffled_displacements[:, layer_idx]               # (n_heldout, d)

        # Fit Procrustes rotation
        R, P_mean = fit_orthogonal_procrustes(P_train, D_train)

        # Fit ridge, selecting alpha by mean CV cosine similarity
        ridge = RidgeCV(
            alphas=ALPHA_GRID, fit_intercept=False, cv=ridge_cv_splitter, scoring=COSINE_SCORER,
        ).fit(P_train, D_train)
        best_alpha = ridge.alpha_

        # Mean CV cosine similarity at best_alpha -- used below to pick the best layer.
        ridge_cv_cos = float(ridge.best_score_)

        # Fit a 2-layer MLP: a more expressive comparison point than the global
        # ridge/orthogonal maps. Cross-validates weight_decay and n_epochs over cv_folds
        # by mean cosine similarity, mirroring RidgeCV's alpha
        # search, then refits on the full train split at the winning hyperparameters.
        mlp, best_mlp_wd, best_mlp_epochs, mlp_cv_cos, mlp_fit_cos = fit_mlp(
            P_train, D_train,
            args.mlp_hidden_dim, device,
            args.mlp_max_epochs, args.mlp_lr, args.mlp_patience,
            MLP_WD_GRID, args.cv_folds, args.seed,
        )

        # SVD of the ridge map for later analysis
        k_svd = min(args.n_svd_components, len(train_record_indices))
        sigma_layer, top_svecs = ridge_svd(ridge, k=k_svd)   # top_svecs: (k, d) left SVs — input directions
        all_sigmas.append(sigma_layer)
        all_top_svecs.append(top_svecs)

        # ── Evaluate on held-out support sets ─────────────────────────────────
        pred_procrustes = predict_procrustes(P_heldout, R, P_mean)
        pred_ridge      = ridge.predict(P_heldout)
        pred_mlp        = predict_mlp(mlp, P_heldout, device)
        if external_A is not None:
            # D = P @ A, matching ridge_svd's convention for the map's direction.
            pred_external = P_heldout @ external_A[layer_idx]

        # Held-out cosine similarity, for MLP.
        mlp_heldout_cos = float(np.mean(batch_cosine(pred_mlp, D_heldout)))

        hyperparameter_records.append({
            "layer": layer_idx,
            "best_alpha": best_alpha,
            "ridge_cv_cos": ridge_cv_cos,
            "best_mlp_wd": best_mlp_wd,
            "best_mlp_epochs": best_mlp_epochs,
            "mlp_cv_cos": mlp_cv_cos,
            "mlp_fit_cos": mlp_fit_cos,
            "mlp_heldout_cos": mlp_heldout_cos,
        })

        # Shuffled-prototype control: randomly permute held-out prototypes before mapping.
        # This breaks the pairing between each prototype and its actual displacement.
        # If the maps produce non-zero cosine similarity even for shuffled prototypes,
        # that would indicate systematic bias (e.g., all displacements pointing the same way).
        shuffle_order         = rng.permutation(len(P_heldout))
        P_heldout_shuffled    = P_heldout[shuffle_order]
        pred_proc_shuffled    = predict_procrustes(P_heldout_shuffled, R, P_mean)
        pred_ridge_shuffled   = ridge.predict(P_heldout_shuffled)
        pred_mlp_shuffled     = predict_mlp(mlp, P_heldout_shuffled, device)
        if external_A is not None:
            pred_external_shuffled = P_heldout_shuffled @ external_A[layer_idx]

        # Cosine similarities between each prediction and the ACTUAL displacement
        cos_procrustes      = batch_cosine(pred_procrustes,     D_heldout)
        cos_ridge           = batch_cosine(pred_ridge,          D_heldout)
        cos_mlp             = batch_cosine(pred_mlp,            D_heldout)
        cos_identity        = batch_cosine(P_heldout,           D_heldout)  # baseline: no map
        cos_proc_shuffled   = batch_cosine(pred_proc_shuffled,  D_heldout)
        cos_ridge_shuffled  = batch_cosine(pred_ridge_shuffled, D_heldout)
        cos_mlp_shuffled    = batch_cosine(pred_mlp_shuffled,   D_heldout)
        if external_A is not None:
            cos_external          = batch_cosine(pred_external,          D_heldout)
            cos_external_shuffled = batch_cosine(pred_external_shuffled, D_heldout)

        # Noise ceiling: same token set, but a random permutation decides the order.
        # Since the prototype fed to every predictor is order/identity-invariant (mean over the set), 
        # this bounds the best any predictor could do.
        cos_noise_ceiling   = batch_cosine(D_heldout_shuffled,  D_heldout)

        mean_cos_ridge_by_layer.append(float(np.mean(cos_ridge)))
        ridge_cv_cos_by_layer.append(ridge_cv_cos)
        all_ridge_models.append(ridge)
        if external_A is not None:
            mean_cos_external_by_layer.append(float(np.mean(cos_external)))

        for j, idx in enumerate(heldout_record_indices):
            rec = records.loc[idx]
            row = {
                "category":                rec["category"],
                "support_id":              rec["support_id"],
                "layer":                   layer_idx,
                "best_alpha":              best_alpha,
                "cos_identity":            float(cos_identity[j]),
                "cos_procrustes":          float(cos_procrustes[j]),
                "cos_ridge":               float(cos_ridge[j]),
                "cos_mlp":                 float(cos_mlp[j]),
                "cos_procrustes_shuffled": float(cos_proc_shuffled[j]),
                "cos_ridge_shuffled":      float(cos_ridge_shuffled[j]),
                "cos_mlp_shuffled":        float(cos_mlp_shuffled[j]),
                "cos_noise_ceiling":       float(cos_noise_ceiling[j]),
            }
            if external_A is not None:
                row["cos_external_A"]          = float(cos_external[j])
                row["cos_external_A_shuffled"] = float(cos_external_shuffled[j])
            output_rows.append(row)

    # ── Save results ──────────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_root, model_name_safe, "displacement_regression")
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(output_rows).to_csv(os.path.join(save_dir, "regressions.csv"), index=False)
    pd.DataFrame(hyperparameter_records).to_csv(os.path.join(save_dir, "hyperparameter_records.csv"), index=False)

    np.save(os.path.join(save_dir, "singular_values.npy"), np.stack(all_sigmas))  # (n_layers, d)
    np.save(os.path.join(save_dir, "top_input_svecs.npy"), np.stack(all_top_svecs))  # (n_layers, k, d)

    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Best layer, by mean 5-fold CV ridge cosine similarity (train split only), 
    # and its full A matrix -- this is what behavioral_modeling.py consumes.
    best_layer = int(np.argmax(ridge_cv_cos_by_layer))
    best_A = all_ridge_models[best_layer].coef_.T   # (d, d)
    np.save(os.path.join(save_dir, "best_A.npy"), best_A)
    with open(os.path.join(save_dir, "best_layer.json"), "w") as f:
        json.dump({
            "best_layer": best_layer,
            "metric": "ridge_cv_cos",
            "value": ridge_cv_cos_by_layer[best_layer],
            "heldout_cos_ridge_at_best_layer": mean_cos_ridge_by_layer[best_layer],
        }, f, indent=2)

    if args.save_all_A:
        all_A = np.stack([model.coef_.T for model in all_ridge_models])  # (n_layers, d, d)
        np.save(os.path.join(save_dir, "all_A.npy"), all_A)
        print(f"Saved all {n_layers} layers' A matrices to {os.path.join(save_dir, 'all_A.npy')}")

    print(f"Saved {len(output_rows)} rows to {save_dir}")
    print(f"Best ridge alphas:\n{pd.DataFrame(hyperparameter_records)[['layer','best_alpha']].to_string(index=False)}")
    print(f"Best layer (mean 5-fold CV cos_ridge): {best_layer} ({ridge_cv_cos_by_layer[best_layer]:.4f}), "
          f"held-out cos_ridge at this layer: {mean_cos_ridge_by_layer[best_layer]:.4f}")
    if external_A is not None:
        print(f"Held-out cos_external_A at this layer: {mean_cos_external_by_layer[best_layer]:.4f} "
              f"(map loaded from {args.external_A_path})")

    # ── Leave-one-concept-out (best layer only) ─────────────────────────────────
    # For each concept, refit the ridge map, Procrustes rotation, and MLP on every OTHER
    # concept's train rows, then evaluate on this concept's held-out rows -- checking
    # whether each map generalizes to a wholly unseen concept, not just unseen support
    # sets of concepts it did see.
    if args.leave_one_concept_out:
        D_train_best   = displacements[train_record_indices, best_layer]
        D_heldout_best = displacements[heldout_record_indices, best_layer]

        # Noise ceiling per concept
        D_heldout_shuffled_best = shuffled_displacements[:, best_layer]   # (n_heldout, d)

        # All categories should be present in both the train and held-out sets.
        train_categories   = records.loc[train_record_indices, "category"].to_numpy()
        heldout_categories = records.loc[heldout_record_indices, "category"].to_numpy()

        loco_categories = sorted(set(train_categories) & set(heldout_categories))

        loco_rows = []
        loco_A_list = []
        for cat in tqdm(loco_categories, desc="Leave-one-concept-out"):
            # Fit a ridge map, a Procrustes rotation, and an MLP on every OTHER concept's train rows
            fit_mask = train_categories != cat
            ridge_loco = RidgeCV(
                alphas=ALPHA_GRID, fit_intercept=False, cv=ridge_cv_splitter, scoring=COSINE_SCORER,
            ).fit(P_train[fit_mask], D_train_best[fit_mask])
            R_loco, P_mean_loco = fit_orthogonal_procrustes(P_train[fit_mask], D_train_best[fit_mask])
            mlp_loco, best_mlp_wd_loco, best_mlp_epochs_loco, _, _ = fit_mlp(
                P_train[fit_mask], D_train_best[fit_mask],
                args.mlp_hidden_dim, device,
                args.mlp_max_epochs, args.mlp_lr, args.mlp_patience,
                MLP_WD_GRID, args.cv_folds, args.seed,
            )

            test_mask = heldout_categories == cat
            pred_ridge_loco = ridge_loco.predict(P_heldout[test_mask]) # Predict the held-out support set of this concept.
            pred_proc_loco  = predict_procrustes(P_heldout[test_mask], R_loco, P_mean_loco)
            pred_mlp_loco   = predict_mlp(mlp_loco, P_heldout[test_mask], device)
            cos_ridge_loco  = batch_cosine(pred_ridge_loco, D_heldout_best[test_mask])
            cos_proc_loco   = batch_cosine(pred_proc_loco,  D_heldout_best[test_mask])
            cos_mlp_loco    = batch_cosine(pred_mlp_loco,   D_heldout_best[test_mask])
            cos_noise_ceiling_loco = batch_cosine(
                D_heldout_shuffled_best[test_mask], D_heldout_best[test_mask]
            )

            for j, idx in enumerate(np.where(test_mask)[0]):
                rec = records.loc[heldout_record_indices[idx]]
                loco_rows.append({
                    "category":              cat,
                    "support_id":            rec["support_id"],
                    "layer":                 best_layer,
                    "best_alpha":            ridge_loco.alpha_,
                    "best_mlp_wd":           best_mlp_wd_loco,
                    "best_mlp_epochs":       best_mlp_epochs_loco,
                    "cos_ridge_loco":        float(cos_ridge_loco[j]),
                    "cos_procrustes_loco":   float(cos_proc_loco[j]),
                    "cos_mlp_loco":          float(cos_mlp_loco[j]),
                    "cos_noise_ceiling_loco": float(cos_noise_ceiling_loco[j]),
                })
            loco_A_list.append(ridge_loco.coef_.T)   # (d, d)

        pd.DataFrame(loco_rows).to_csv(os.path.join(save_dir, "loco_regressions.csv"), index=False)
        np.save(os.path.join(save_dir, "loco_A.npy"), np.stack(loco_A_list))  # (n_concepts, d, d)
        with open(os.path.join(save_dir, "loco_categories.json"), "w") as f:
            json.dump(loco_categories, f, indent=2)

        mean_loco_cos      = float(np.mean([r["cos_ridge_loco"] for r in loco_rows]))
        mean_loco_proc_cos = float(np.mean([r["cos_procrustes_loco"] for r in loco_rows]))
        mean_loco_mlp_cos  = float(np.mean([r["cos_mlp_loco"] for r in loco_rows]))
        mean_loco_nc       = float(np.mean([r["cos_noise_ceiling_loco"] for r in loco_rows]))
        print(f"Leave-one-concept-out (layer {best_layer}): mean cos_ridge = {mean_loco_cos:.4f}, "
              f"mean cos_procrustes = {mean_loco_proc_cos:.4f}, mean cos_mlp = {mean_loco_mlp_cos:.4f}, "
              f"mean cos_noise_ceiling = {mean_loco_nc:.4f} over {len(loco_categories)} concepts")
