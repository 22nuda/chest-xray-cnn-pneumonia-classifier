"""
simple_cnn.py
--------------
Educational Kaggle computer vision project:
Binary CNN classifier for chest X-ray images (PNEUMONIA vs NORMAL) using TensorFlow/Keras.

Features:
- Loads JPG/PNG/BMP from train/val/test OR from preprocessed train_mod/val_mod/test_mod
- Optional preprocessing mode: resize to IMG_SIZE and cache images as BMP in *_mod folders
- CNN with augmentation, BatchNorm, L2 regularization, Dropout, GlobalAveragePooling
- Trains with EarlyStopping + ModelCheckpoint monitored by validation AUC
- Finds best decision threshold on validation set by maximizing F1 for the PNEUMONIA class
- Evaluates on test set and prints confusion matrix + classification report
- Saves training curves and the best model (.keras)

Expected directory structure (dataset NOT included in this repo):
  ./train/NORMAL, ./train/PNEUMONIA
  ./val/NORMAL,   ./val/PNEUMONIA
  ./test/NORMAL,  ./test/PNEUMONIA

Optional cached/preprocessed structure:
  ./train_mod/NORMAL, ./train_mod/PNEUMONIA
  ./val_mod/NORMAL,   ./val_mod/PNEUMONIA
  ./test_mod/NORMAL,  ./test_mod/PNEUMONIA
"""

from __future__ import annotations

import os
import gc
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Default configuration
# -----------------------------
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # label 0 and 1
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Config:
    seed: int = 42
    img_size: int = 256
    batch_size: int = 4
    epochs: int = 15

    # If True, compute and use class weights to reduce class imbalance impact
    use_class_weight: bool = False

    # L2 regularization strength for Conv2D kernels
    l2_reg: float = 1e-4

    # Limit number of images per class (None = all)
    max_train_per_class: Optional[int] = None
    max_val_per_class: Optional[int] = None
    max_test_per_class: Optional[int] = None

    # Mode:
    # True  -> read from train/val/test, resize + normalize, and cache as BMP to *_mod
    # False -> read directly from train_mod/val_mod/test_mod (already cached)
    save_bmp_cache: bool = False

    # Files / folders
    best_model_name: str = "pneumonia_cnn_best.keras"
    artifacts_dir_name: str = "artifacts"


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    """Prevents TF from allocating all GPU memory upfront (useful on smaller VRAM)."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print("[INFO] Using GPU(s):", gpus)
        else:
            print("[INFO] No GPU detected, running on CPU.")
    except Exception as e:
        print("[WARN] GPU configuration error:", e)


def ensure_class_folders_exist(base_dir: str) -> None:
    """Validate that base_dir/{NORMAL,PNEUMONIA} exist."""
    for cname in CLASS_NAMES:
        p = os.path.join(base_dir, cname)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"[ERROR] Missing class folder: {p}")


def list_image_files(folder: str, max_count: Optional[int]) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXT)]
    files.sort()
    if max_count is not None:
        files = files[:max_count]
    return files


def load_images_from_dir(
    src_base_dir: str,
    dst_cache_dir: str,
    img_size: int,
    max_per_class: Optional[int],
    save_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from src_base_dir/{NORMAL,PNEUMONIA}.
    Optionally save resized images as BMP into dst_cache_dir/{NORMAL,PNEUMONIA}.

    Returns:
      X: float32 array (N, img_size, img_size, 3) in [0,1]
      y: int64 array (N,) labels {0,1}
    """
    ensure_class_folders_exist(src_base_dir)

    X: List[np.ndarray] = []
    y: List[int] = []

    for class_name, label in zip(CLASS_NAMES, [0, 1]):
        src_folder = os.path.join(src_base_dir, class_name)
        files = list_image_files(src_folder, max_per_class)

        total = len(files)
        print(f"[INFO] Loading {total} images from {src_folder} as {class_name} ({label})")

        # Cache folder (only used if save_cache=True)
        dst_label_folder = os.path.join(dst_cache_dir, class_name)
        if save_cache:
            os.makedirs(dst_label_folder, exist_ok=True)

        for i, fname in enumerate(files, start=1):
            src_path = os.path.join(src_folder, fname)

            # Decode with TF (fast + supports JPEG/PNG/BMP)
            try:
                img_raw = tf.io.read_file(src_path)
                img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, (img_size, img_size))  # float32 approx [0..255]
            except Exception as e:
                print(f"[WARN] Skipping unreadable image: {src_path} ({e})")
                continue

            # Optional: save BMP cache for faster repeated experiments
            if save_cache:
                img_uint8 = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8).numpy()
                pil_img = Image.fromarray(img_uint8, mode="RGB")

                out_name = os.path.splitext(fname)[0] + ".bmp"
                dst_path = os.path.join(dst_label_folder, out_name)
                pil_img.save(dst_path, format="BMP")

                if i % 200 == 0 or i == total:
                    print(f"[CACHE] {class_name}: {i}/{total} saved -> {dst_path}")

            # Normalize to [0,1] for training
            img_norm = (img / 255.0).numpy().astype(np.float32)
            X.append(img_norm)
            y.append(label)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Loaded X shape: {X_arr.shape}, y shape: {y_arr.shape}")
    return X_arr, y_arr


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced weights: total/(2*n_class)."""
    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    total = n0 + n1

    w0 = total / (2.0 * n0) if n0 > 0 else 1.0
    w1 = total / (2.0 * n1) if n1 > 0 else 1.0

    print(f"[INFO] Train counts -> NORMAL: {n0}, PNEUMONIA: {n1}, TOTAL: {total}")
    print(f"[INFO] Class weights -> 0: {w0:.3f}, 1: {w1:.3f}")
    return {0: float(w0), 1: float(w1)}


def build_model(img_size: int, l2_reg: float) -> tf.keras.Model:
    """CNN with augmentation + BatchNorm + L2 + GAP."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)

    for filters in [128, 256, 512, 256, 128]:
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # binary probability

    model = models.Model(inputs, outputs, name="pneumonia_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


class SimpleLogger(tf.keras.callbacks.Callback):
    """Clean, readable per-epoch logging."""
    def on_train_begin(self, logs=None):
        epochs = self.params.get("epochs", None)
        print(f"\n[TRAIN] Starting training ({epochs} epochs)\n")

    def on_epoch_begin(self, epoch, logs=None):
        epochs = self.params.get("epochs", None)
        print(f"\n[TRAIN] ===== Epoch {epoch + 1}/{epochs} =====")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"[TRAIN] loss={logs.get('loss', float('nan')):.4f}, "
            f"acc={logs.get('accuracy', float('nan')):.4f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.4f}, "
            f"val_acc={logs.get('val_accuracy', float('nan')):.4f}, "
            f"val_auc={logs.get('val_auc', float('nan')):.4f}"
        )


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Scan thresholds and pick the one maximizing F1 for the positive class (PNEUMONIA=1).
    Returns: (best_threshold, best_f1)
    """
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.10, 0.90, 81):  # step 0.01
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, float(best_f1)


def save_history_plot(
    history: tf.keras.callbacks.History,
    key: str,
    val_key: str,
    title: str,
    out_path: str,
) -> None:
    if key not in history.history or val_key not in history.history:
        return
    plt.figure()
    plt.plot(history.history[key], label="train")
    plt.plot(history.history[val_key], label="val")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def main() -> None:
    cfg = Config()

    set_reproducibility(cfg.seed)
    configure_gpu_memory_growth()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, cfg.artifacts_dir_name)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Raw dataset folders
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Cached/preprocessed folders
    train_mod_dir = os.path.join(base_dir, "train_mod")
    val_mod_dir = os.path.join(base_dir, "val_mod")
    test_mod_dir = os.path.join(base_dir, "test_mod")

    # Create cache folders (ok even if unused)
    os.makedirs(train_mod_dir, exist_ok=True)
    os.makedirs(val_mod_dir, exist_ok=True)
    os.makedirs(test_mod_dir, exist_ok=True)

    # Decide data source based on caching mode
    if cfg.save_bmp_cache:
        src_train_dir, src_val_dir, src_test_dir = train_dir, val_dir, test_dir
        print("[INFO] MODE: Preprocess raw images -> cache BMP into *_mod")
    else:
        src_train_dir, src_val_dir, src_test_dir = train_mod_dir, val_mod_dir, test_mod_dir
        print("[INFO] MODE: Use cached images from *_mod (no new caching)")

    # Basic checks: root folders must exist
    for p in [src_train_dir, src_val_dir, src_test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"[ERROR] Source directory not found: {p}")

    print("[INFO] Base directory: ", base_dir)
    print("[INFO] Source TRAIN:   ", src_train_dir)
    print("[INFO] Source VAL:     ", src_val_dir)
    print("[INFO] Source TEST:    ", src_test_dir)
    print("[INFO] Artifacts dir:  ", artifacts_dir)

    # -----------------------------
    # Load data (and optionally cache BMP)
    # -----------------------------
    print("\n[INFO] Loading TRAIN data...")
    x_train_raw, y_train_raw = load_images_from_dir(
        src_train_dir, train_mod_dir, cfg.img_size, cfg.max_train_per_class, cfg.save_bmp_cache
    )

    print("\n[INFO] Loading VAL data...")
    x_val_raw, y_val_raw = load_images_from_dir(
        src_val_dir, val_mod_dir, cfg.img_size, cfg.max_val_per_class, cfg.save_bmp_cache
    )

    print("\n[INFO] Loading TEST data...")
    x_test, y_test = load_images_from_dir(
        src_test_dir, test_mod_dir, cfg.img_size, cfg.max_test_per_class, cfg.save_bmp_cache
    )

    # -----------------------------
    # Create a fresh train/val split from (train + val)
    # -----------------------------
    X_all = np.concatenate([x_train_raw, x_val_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.10,
        random_state=cfg.seed,
        stratify=y_all,
    )

    # Shuffle training set
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # Optional class weights
    class_weight = compute_class_weights(y_train) if cfg.use_class_weight else None
    if class_weight is None:
        print("[INFO] Class weights disabled (use_class_weight=False).")

    # -----------------------------
    # Build + train model
    # -----------------------------
    print("\n[INFO] Building model...")
    model = build_model(cfg.img_size, cfg.l2_reg)
    model.summary()

    best_model_path = os.path.join(artifacts_dir, cfg.best_model_name)

    callbacks = [
        SimpleLogger(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    print("\n[INFO] Starting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # -----------------------------
    # Free memory + load best model
    # -----------------------------
    print("\n[INFO] Loading best model:", best_model_path)
    del model
    gc.collect()
    K.clear_session()

    best_model = tf.keras.models.load_model(best_model_path)

    # -----------------------------
    # Threshold tuning on validation set (maximize F1 for PNEUMONIA)
    # -----------------------------
    print("\n[INFO] Searching best threshold on validation set (maximize F1 for PNEUMONIA)...")
    val_probs = best_model.predict(x_val, verbose=0).ravel()
    best_thr, best_f1 = find_best_threshold_by_f1(y_val.astype(int), val_probs)

    print(f"[INFO] Best threshold (VAL, by F1): {best_thr:.3f} | F1={best_f1:.4f}")

    val_cm = confusion_matrix(y_val.astype(int), (val_probs >= best_thr).astype(int))
    print("\nValidation confusion matrix (best threshold):\n", val_cm)

    # -----------------------------
    # Evaluate on test set
    # -----------------------------
    print("\n[INFO] Evaluating on TEST data (Keras metrics at threshold=0.5)...")
    test_loss, test_acc, test_auc = best_model.evaluate(x_test, y_test, verbose=1)
    print(f"\n[RESULT] Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

    print("\n[INFO] Test confusion matrix + report (using best threshold from VAL)...")
    y_prob = best_model.predict(x_test, verbose=0).ravel()
    y_pred = (y_prob >= best_thr).astype(int)
    y_true = y_test.astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion matrix (TEST, best_thr={best_thr:.3f}):\n{cm}")

    print(f"\nClassification report (TEST, best_thr={best_thr:.3f}):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # -----------------------------
    # Save training curves
    # -----------------------------
    save_history_plot(
        history, "accuracy", "val_accuracy",
        "Accuracy", os.path.join(artifacts_dir, "training_accuracy.png")
    )
    save_history_plot(
        history, "loss", "val_loss",
        "Loss", os.path.join(artifacts_dir, "training_loss.png")
    )
    save_history_plot(
        history, "auc", "val_auc",
        "AUC", os.path.join(artifacts_dir, "training_auc.png")
    )

    print("\n[DONE] Best model saved to:", best_model_path)
    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
