"""
STEP 3: Trainer – optimized for maximum accuracy (4 labels: TAP, WAVE, SHAKE, IDEAL)

MLOps Lesson 1: Reads config from config.yaml (random_seed, epochs, lr, etc.)
so every run is reproducible and documented.

Training rules (AI/ML best practice):
- Reproducibility: fixed random seeds (numpy, TensorFlow).
- No data leakage: train and test sets have no overlapping files (validated).
- Normalization: fitted on training data only; validation/test use same layer.
- Validation: stratified 85/15 train/val so each class is represented in val.
- Class balance: inverse-frequency class weights + extra weight for SHAKE.
- Regularization: L2, dropout; early stopping on val_loss; LR reduction on plateau.
- Test set: used only for final metrics; never for training or model selection.

- Input: 39 features, normalized (layer baked into model for TFLite)
- Architecture: Normalize → Dense(20) → Dropout → Dense(10) → Dropout → Dense(4, softmax)
- Saves SavedModel for TFLite (Step 4)
"""

import numpy as np
from pathlib import Path

from dataset import (
    get_training_dataset,
    get_testing_dataset,
    LABEL_ORDER,
    index_to_label,
)
from data_loader import load_training_data, load_testing_data, validate_all_data
from mlops_config import load_config, get_project_root

import tensorflow as tf

# Defaults (overridden by config.yaml when present)
RANDOM_SEED = 42
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"


def build_model(input_dim=39, num_classes=4, l2=1e-4, dropout=0.2):
    """
    Matches your screenshot: Input(39) → Normalize → Dense(20) → Dropout → Dense(10) → Dropout → Dense(4, softmax).
    """
    reg = tf.keras.regularizers.L2(l2)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Normalization(axis=-1),  # will adapt in fit()
        tf.keras.layers.Dense(20, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout * 0.8),
        tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg),
    ])
    return model


def compute_class_weights(y):
    """Inverse frequency weights. Only SHAKE (index 2) gets extra 2x so it's not missed on device."""
    counts = np.bincount(y, minlength=len(LABEL_ORDER))
    total = len(y)
    weights = total / (len(LABEL_ORDER) * np.maximum(counts, 1))
    w = dict(enumerate(weights.astype(np.float32)))
    w[2] = float(w.get(2, 1.0) * 2.0)   # SHAKE
    return w


def train_and_evaluate(epochs=250, verbose=1, config=None, save_dir=None):
    """
    Train with normalization, class weights, early stopping, and LR schedule.
    Reports per-class accuracy and weighted F1.
    Enforces: reproducibility (seeds), no train/test leakage, stratified validation.
    If config is provided (from config.yaml), uses it for seed, epochs, lr, etc.
    If save_dir is provided (e.g. runs/run_123/saved_model), SavedModel is written there for MLOps versioning.
    Returns: (model, test_accuracy, test_f1).
    """
    cfg = config or {}
    seed = cfg.get("random_seed", RANDOM_SEED)
    epochs = cfg.get("epochs", epochs)
    lr = cfg.get("learning_rate", 0.001)
    val_split = cfg.get("train_val_split", 0.15)
    es_patience = cfg.get("early_stopping_patience", 35)
    rl_patience = cfg.get("reduce_lr_patience", 12)
    rl_factor = cfg.get("reduce_lr_factor", 0.5)
    min_lr = cfg.get("min_lr", 1e-6)
    input_dim = cfg.get("input_dim", 39)
    num_classes_cfg = cfg.get("num_classes", 4)
    l2 = cfg.get("l2", 1e-4)
    dropout = cfg.get("dropout", 0.2)
    saved_model_dir = cfg.get("saved_model_dir", "saved_model")
    model_dir = get_project_root() / saved_model_dir

    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    raw_train = load_training_data()
    raw_test = load_testing_data()

    # Data validation: leakage and label coverage
    validation_ok, _ = validate_all_data(raw_train, raw_test, LABEL_ORDER, verbose=verbose)
    if not validation_ok:
        raise RuntimeError("Data validation failed (e.g. train/test leakage). Fix data and re-run.")

    rec_counts = {}
    for r in raw_train:
        rec_counts[r["label"]] = rec_counts.get(r["label"], 0) + 1
    for lbl in LABEL_ORDER:
        n = rec_counts.get(lbl, 0)
        if n < 5 and verbose:
            print(f"  Tip: Only {n} {lbl} recording(s). Add more Training/{lbl.lower()}.*.json for better on-device recognition.")

    X_train, y_train, _ = get_training_dataset()
    X_test, y_test, _ = get_testing_dataset()

    if verbose:
        print(f"Data: {len(X_train)} train windows, {len(X_test)} test windows (before val split).")

    # Stratified train/val split so validation reflects class distribution
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_split, stratify=y_train, random_state=seed
        )
        validation_data = (X_val, y_val)
        if verbose:
            print(f"Stratified split: train {len(X_train)}, val {len(X_val)}")
    except Exception:
        validation_data = None  # fallback to validation_split
        if verbose:
            print(f"Using Keras validation_split={val_split} (stratified split unavailable)")

    num_classes = len(LABEL_ORDER)
    model = build_model(input_dim=X_train.shape[1], num_classes=num_classes, l2=l2, dropout=dropout)

    # Adapt normalization layer on training data only
    norm_layer = model.layers[0]
    norm_layer.adapt(X_train)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(y_train)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=rl_factor,
            patience=rl_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

    fit_kw = dict(
        epochs=epochs,
        verbose=verbose,
        shuffle=True,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    if validation_data is not None:
        fit_kw["validation_data"] = validation_data
    else:
        fit_kw["validation_split"] = val_split

    history = model.fit(X_train, y_train, **fit_kw)

    # Test evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.2%}  (Test loss: {test_loss:.4f})")

    # Per-class metrics
    pred_probs = model.predict(X_test, verbose=0)
    pred_indices = np.argmax(pred_probs, axis=1)
    print("\nPer-class accuracy (test set):")
    for i, label in enumerate(LABEL_ORDER):
        mask = y_test == i
        if mask.sum() == 0:
            print(f"  {label}: no samples")
            continue
        acc = (pred_indices[mask] == i).mean()
        print(f"  {label}: {acc:.1%} ({int(mask.sum())} samples)")
        if acc < 0.9 and verbose:
            print(f"    -> For better {label} accuracy, add more varied Training/{label.lower()}.*.json and retrain.")

    # Weighted F1 (sklearn-style) – returned for MLOps experiment tracking
    test_f1 = 0.0
    try:
        from sklearn.metrics import f1_score
        test_f1 = float(f1_score(y_test, pred_indices, average="weighted", zero_division=0))
        print(f"\nWeighted F1 (test): {test_f1:.3f}")
    except Exception:
        pass

    print("\nTest set predictions (first 20):")
    n_show = min(20, len(y_test))
    for i in range(n_show):
        true_label = index_to_label(int(y_test[i]))
        pred_label = index_to_label(int(pred_indices[i]))
        ok = "[OK]" if pred_indices[i] == y_test[i] else "[X]"
        print(f"  {i+1}: true={true_label}, predicted={pred_label} {ok}")
    if len(y_test) > n_show:
        print(f"  ... and {len(y_test) - n_show} more")

    # Optional: save to a different dir (e.g. runs/run_<id>/saved_model) for MLOps versioning
    actual_save_dir = Path(save_dir) if save_dir else model_dir
    actual_save_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(actual_save_dir))
    print(f"\nModel saved to: {actual_save_dir}")

    return model, float(test_acc), test_f1


if __name__ == "__main__":
    print("Step 3: Trainer – max accuracy for 4 labels (TAP, WAVE, SHAKE, IDEAL).")
    print("MLOps: using config.yaml for seed, epochs, lr, etc.\n")
    try:
        cfg = load_config()
        print(f"Config loaded: epochs={cfg.get('epochs')}, lr={cfg.get('learning_rate')}, seed={cfg.get('random_seed')}")
    except FileNotFoundError:
        cfg = {}
        print("No config.yaml found; using defaults.")
    train_and_evaluate(verbose=1, config=cfg)
    print("\nDone. Run export_tflite.py, then copy model to app assets and build APK.")
