# Simple_CNN.py
# CNN klasifikátor RTG pľúc (PNEUMONIA vs NORMAL)
# - načíta JPG/PNG/BMP z train/val/test alebo train_mod/val_mod/test_mod
# - pri SAVE_BMP=True: dekóduje + zmení veľkosť na IMG_SIZE x IMG_SIZE a uloží BMP do *_mod
# - trénuje model na normalizovaných numpy poliach
#
# Kompatibilné s TensorFlow-GPU 2.10 (CUDA 11.2, cuDNN 8.1)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ak VRAM nestačí - pri -1 beží na CPU

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image  # na ukladanie BMP
import gc
from tensorflow.keras import backend as K

# ============================
# ZÁKLADNÉ NASTAVENIA
# ============================
SEED        = 42
IMG_SIZE    = 256           # cieľový rozmer obrázkov
BATCH_SIZE  = 4
EPOCHS      = 15

# prepínač: používať class_weight pri tréningu?
USE_CLASS_WEIGHT = False     # <--- MÔŽEŠ PREPNÚŤ NA True AK CHCEŠ

# L2 regularizácia pre Conv vrstvy (proti preučeniu)
L2_REG      = 1e-4

# koľko obrázkov zobrať z každého priečinka NORMAL / PNEUMONIA
# None = zober všetky
MAX_TRAIN_PER_CLASS = None
MAX_VAL_PER_CLASS   = None
MAX_TEST_PER_CLASS  = None

# Režim:
#   SAVE_BMP = True  -> zdroj = train/val/test, urobí preprocessing + uloží BMP do *_mod
#   SAVE_BMP = False -> zdroj = train_mod/val_mod/test_mod, už len načítava hotové BMP
SAVE_BMP = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# GPU NASTAVENIE
# ============================
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[INFO] Using GPU(s):", gpus)
    else:
        print("[INFO] No GPU detected, running on CPU.")
except Exception as e:
    print("[WARN] GPU config error:", e)

# ============================
# CESTY K DÁTAM
# ============================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# originálne dáta
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# priečinky pre UPRAVENÉ obrázky (BMP)
TRAIN_MOD_DIR = os.path.join(BASE_DIR, "train_mod")
VAL_MOD_DIR   = os.path.join(BASE_DIR, "val_mod")
TEST_MOD_DIR  = os.path.join(BASE_DIR, "test_mod")

MODEL_BEST_PATH = os.path.join(BASE_DIR, "pneumonia_cnn_best_full.keras")

print("[INFO] BASE_DIR:      ", BASE_DIR)
print("[INFO] TRAIN_DIR:     ", TRAIN_DIR)
print("[INFO] VAL_DIR:       ", VAL_DIR)
print("[INFO] TEST_DIR:      ", TEST_DIR)
print("[INFO] TRAIN_MOD_DIR: ", TRAIN_MOD_DIR)
print("[INFO] VAL_MOD_DIR:   ", VAL_MOD_DIR)
print("[INFO] TEST_MOD_DIR:  ", TEST_MOD_DIR)

# vytvor mod priečinky (ak už existujú, nič sa nedeje)
os.makedirs(TRAIN_MOD_DIR, exist_ok=True)
os.makedirs(VAL_MOD_DIR,   exist_ok=True)
os.makedirs(TEST_MOD_DIR,  exist_ok=True)

# podľa SAVE_BMP vyberieme zdroj
if SAVE_BMP:
    SRC_TRAIN_DIR = TRAIN_DIR
    SRC_VAL_DIR   = VAL_DIR
    SRC_TEST_DIR  = TEST_DIR
    print("[INFO] MODE: PREPROCESS ORIGINÁLOV -> ukladám BMP do *_mod")
else:
    SRC_TRAIN_DIR = TRAIN_MOD_DIR
    SRC_VAL_DIR   = VAL_MOD_DIR
    SRC_TEST_DIR  = TEST_MOD_DIR
    print("[INFO] MODE: POUŽÍVAM UŽ HOTOVÉ BMP z *_mod (žiadne nové ukládanie)")

# skontroluj, že zdrojové dátové priečinky existujú
for p in [SRC_TRAIN_DIR, SRC_VAL_DIR, SRC_TEST_DIR]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"[ERROR] Source directory not found: {p}")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # label 0 a 1

# ============================
# FUNKCIA: NAČÍTANIE + ULOŽENIE BMP (ak SAVE_BMP=True)
# ============================
def load_subset_from_dir(src_base_dir, dst_base_dir, img_size, max_per_class):
    """
    src_base_dir:  train/val/test alebo train_mod/val_mod/test_mod (zdroj obrázkov)
    dst_base_dir:  train_mod/val_mod/test_mod (kam ukladáme BMP, ak SAVE_BMP=True)
    img_size:      cieľový rozmer (img_size x img_size)
    max_per_class: koľko obrázkov zobrať z každej triedy (None = všetky)

    Výstup:
      images: numpy array tvaru (N, img_size, img_size, 3), hodnoty 0–1
      labels: numpy array tvaru (N,), hodnoty 0 alebo 1
    """
    images = []
    labels = []

    for label_name, label in zip(CLASS_NAMES, [0, 1]):
        # zdrojový priečinok (JPG/PNG/BMP)
        src_folder = os.path.join(src_base_dir, label_name)
        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        files.sort()
        if max_per_class is not None:
            files = files[:max_per_class]

        total = len(files)
        print(f"[INFO] Loading {total} images from {src_folder} as label {label_name} ({label})")

        # cieľový priečinok pre BMP (len ak SAVE_BMP=True)
        dst_label_folder = os.path.join(dst_base_dir, label_name)
        if SAVE_BMP:
            os.makedirs(dst_label_folder, exist_ok=True)

        for i, fname in enumerate(files, start=1):
            src_path = os.path.join(src_folder, fname)

            # 1) načítanie a dekódovanie (JPG/PNG/BMP) → tensor (H, W, 3)
            img_raw = tf.io.read_file(src_path)
            img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)

            # 2) resize na (img_size, img_size); TensorFlow spraví float32
            img = tf.image.resize(img, (img_size, img_size))  # float32, rozsah ~0–255

            # 3) ak SAVE_BMP=True -> ulož BMP (0–255, uint8) cez Pillow
            if SAVE_BMP:
                img_uint8 = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
                np_img = img_uint8.numpy()
                pil_img = Image.fromarray(np_img, mode="RGB")

                out_name = os.path.splitext(fname)[0] + ".bmp"
                dst_path = os.path.join(dst_label_folder, out_name)
                pil_img.save(dst_path, format="BMP")

                # log, aby si videl progres
                print(f"[SAVE] {label_name}: {i}/{total} saved -> {dst_path}")

            # 4) normalizácia pre tréning (0–1, float32)
            img_norm = img / 255.0
            images.append(img_norm.numpy())
            labels.append(label)

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    print(f"[INFO] Loaded array shape: {images.shape}, labels shape: {labels.shape}")
    return images, labels

# ============================
# NAČÍTANIE DÁT (A PRÍPADNÉ VYTVORENIE BMP)
# ============================
print("\n[INFO] Loading TRAIN data...")
x_train_raw, y_train_raw = load_subset_from_dir(
    SRC_TRAIN_DIR, TRAIN_MOD_DIR, IMG_SIZE, MAX_TRAIN_PER_CLASS
)

print("\n[INFO] Loading VAL data...")
x_val_raw, y_val_raw = load_subset_from_dir(
    SRC_VAL_DIR, VAL_MOD_DIR, IMG_SIZE, MAX_VAL_PER_CLASS
)

print("\n[INFO] Loading TEST data...")
x_test, y_test = load_subset_from_dir(
    SRC_TEST_DIR, TEST_MOD_DIR, IMG_SIZE, MAX_TEST_PER_CLASS
)

# ============================
# NOVÝ TRAIN/VAL SPLIT ZO VŠETKÝCH TRAIN+VAL DÁT
# ============================
X_all = np.concatenate([x_train_raw, x_val_raw], axis=0)
y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)

x_train, x_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.1,          # 10 % na validáciu
    random_state=SEED,
    stratify=y_all          # zachová pomer NORMAL / PNEUMONIA
)

# premiešanie trénovacích dát
perm = np.random.permutation(len(x_train))
x_train, y_train = x_train[perm], y_train[perm]

# výpočet class weights podľa trénovacej sady (iba ak USE_CLASS_WEIGHT=True)
if USE_CLASS_WEIGHT:
    n0 = np.sum(y_train == 0)
    n1 = np.sum(y_train == 1)
    total = n0 + n1
    class_weight = {
        0: total / (2.0 * n0) if n0 > 0 else 1.0,
        1: total / (2.0 * n1) if n1 > 0 else 1.0,
    }
    print(f"\n[INFO] New Train counts -> NORMAL: {int(n0)}, PNEUMONIA: {int(n1)}, TOTAL: {int(total)}")
    print(f"[INFO] New Class weights -> 0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}")
else:
    class_weight = None
    print("\n[INFO] Class weights disabled (USE_CLASS_WEIGHT=False).")

# ============================
# DEFINÍCIA MODEL (CNN + AUGMENTÁCIA + BatchNorm + L2)
# ============================
print("\n[INFO] Building CNN model...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)

# Konvolučné bloky s BatchNormalization + L2 regularizáciou
for filters in [128, 256, 512, 256, 128]:
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

# Namiesto veľkého Flatten použijeme GlobalAveragePooling2D
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # binárny výstup

model = models.Model(inputs, outputs, name="pneumonia_cnn_full")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# ============================
# CALLBACKY
# ============================
class SimpleLogger(tf.keras.callbacks.Callback):
    """Jednoduchý logger pre prehľadný výpis počas tréningu."""
    def on_train_begin(self, logs=None):
        print("\n[TRAIN] Starting training for {} epochs\n".format(
            self.params.get("epochs", EPOCHS)))

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n[TRAIN] ===== Epoch {epoch+1}/{self.params.get('epochs', EPOCHS)} =====")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"[TRAIN] Epoch {epoch+1} | "
            f"loss={logs.get('loss', float('nan')):.4f}, "
            f"acc={logs.get('accuracy', float('nan')):.4f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.4f}, "
            f"val_acc={logs.get('val_accuracy', float('nan')):.4f}, "
            f"val_auc={logs.get('val_auc', float('nan')):.4f}"
        )

callbacks = [
    SimpleLogger(),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",           # <- DÔLEŽITÉ: hľadáme maximum AUC
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_BEST_PATH,
        monitor="val_auc",
        mode="max",           # <- DÔLEŽITÉ: ukladáme model s max AUC
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",           # <- DÔLEŽITÉ: LR meníme podľa maxima AUC
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    ),
]

# ============================
# TRÉNING
# ============================
print("\n[INFO] Starting model.fit() ...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

# ============================
# UVOĽNENIE PAMÄTE A NAČÍTANIE NAJLEPŠIEHO MODELU + TEST
# ============================
print("\n[INFO] Loading best model from:", MODEL_BEST_PATH)

# uvoľniť pôvodný model z pamäte pred load_model
del model
gc.collect()
K.clear_session()

best_model = tf.keras.models.load_model(MODEL_BEST_PATH)

# ============================
# HĽADANIE OPTIMÁLNEHO THRESHOLDU NA VAL SADA
# ============================
print("\n[INFO] Searching best threshold on VALIDATION set ...")
val_probs = best_model.predict(x_val, verbose=0).ravel()
y_val_int = y_val.astype(int)

best_thr = 0.5
best_f1 = -1.0

for thr in np.linspace(0.1, 0.9, 81):  # kroky po 0.01
    y_val_pred = (val_probs >= thr).astype(int)
    f1 = f1_score(y_val_int, y_val_pred, pos_label=1)  # PNEUMONIA ako pozitívna trieda
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"[INFO] Best threshold on VAL (by F1 for PNEUMONIA): {best_thr:.3f}, F1 = {best_f1:.4f}")

# (voliteľné) môžeš si pozrieť aj confusion matrix na val
val_cm = confusion_matrix(y_val_int, (val_probs >= best_thr).astype(int))
print("\nValidation confusion matrix (using best threshold):\n", val_cm)

# ============================
# VYHODNOTENIE NA TEST DÁTACH (KERAS METRIKY @0.5)
# ============================
print("\n[INFO] Evaluating on TEST data (Keras metrics @0.5 threshold)...")
test_loss, test_acc, test_auc = best_model.evaluate(x_test, y_test, verbose=1)
print(f"\n[RESULT] Test (Keras) -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

# ============================
# CONFUSION MATRIX & REPORT NA TESTE (POUŽIJEME BEST_THR)
# ============================
print("\n[INFO] Calculating confusion matrix and classification report on TEST (with best threshold)...")
y_prob = best_model.predict(x_test, verbose=0).ravel()
y_pred = (y_prob >= best_thr).astype(int)
y_true = y_test.astype(int)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (test, best_thr={:.3f}):\n{}".format(best_thr, cm))

print("\nClassification report (test, best_thr={:.3f}):".format(best_thr))
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ============================
# GRAFY PRIEBEHU TRÉNINGU
# ============================
def plot_history(h, key, val_key, title, fname):
    if key not in h.history:
        return
    plt.figure()
    plt.plot(h.history[key], label="train")
    plt.plot(h.history[val_key], label="val")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] {title} plot saved to: {out_path}")

plot_history(history, "accuracy", "val_accuracy",
             "Accuracy", "training_accuracy_full.png")
plot_history(history, "loss", "val_loss",
             "Loss", "training_loss_full.png")
plot_history(history, "auc", "val_auc",
             "AUC", "training_auc_full.png")

print("\n[DONE] Best model saved as:", MODEL_BEST_PATH)
print("[DONE] Training finished.")
