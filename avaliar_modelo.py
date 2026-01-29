# Avalia um modelo treinado (MobileNetV2 ou ResNet18) em um dataset ImageFolder (val),
# gera métricas + matriz de confusão + lista dos erros mais confiantes.

# Requisitos:
#   pip install torch torchvision pillow matplotlib

import os
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


# CONFIG
CKPT_PATH = r"C:\Data_Science\Aulas\6_Semestre\Deep Learning\Code_final_flores\mobilenetv2_best.pth"
# Ex.: r"...\resnet18_best.pth"


DATA_DIR = r"C:\Data_Science\Aulas\6_Semestre\Deep Learning\Code_final_flores\dados"
VAL_FOLDER = "flores_validation"


BATCH_SIZE = 32
NUM_WORKERS = 2


# Saídas
OUT_DIR = r"C:\Data_Science\Aulas\6_Semestre\Deep Learning\Code_final_flores\avaliacao"
CONFUSION_PNG = "confusion_matrix.png"
ERRORS_TXT = "top_errors.txt"


# Quantos erros mais confiantes salvar
TOP_ERRORS = 30


# Se quiser também testar uma imagem única:
RUN_SINGLE_IMAGE_TEST = False
IMAGE_PATH = r"C:\Data_Science\Aulas\6_Semestre\Deep Learning\Codes_antigos\code_flores_v3\teste.jpg"
TOPK_SINGLE = 5


# UTIL
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def load_checkpoint(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt or "classes" not in ckpt:
        raise KeyError(
            f"Checkpoint precisa conter 'model_state_dict' e 'classes'. Chaves: {list(ckpt.keys())}"
        )
    return ckpt



def build_model_from_ckpt(ckpt: dict):
    arch = ckpt.get("arch", None)
    if arch is None:
        raise KeyError("Checkpoint não tem a chave 'arch'. Salve com 'arch' no treino (mobilenet_v2 ou resnet18).")

    classes = ckpt["classes"]
    num_classes = len(classes)
    img_size = int(ckpt.get("img_size", 224))
    dropout_p = float(ckpt.get("dropout", 0.0))

    arch_l = str(arch).lower()
    if arch_l in ("mobilenet_v2", "mobilenetv2", "mobilenet"):
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        arch_norm = "mobilenet_v2"
    elif arch_l in ("resnet18", "resnet_18"):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        arch_norm = "resnet18"
    else:
        raise ValueError(f"Arquitetura não suportada no ckpt: {arch}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model, arch_norm, classes, img_size, dropout_p



def get_val_transform(img_size: int):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])



def compute_macro_metrics(conf: np.ndarray):
    # conf[true, pred]
    K = conf.shape[0]
    eps = 1e-12

    precisions = []
    recalls = []
    f1s = []
    supports = conf.sum(axis=1)

    for c in range(K):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return macro_p, macro_r, macro_f1, np.array(precisions), np.array(recalls), np.array(f1s), supports



def plot_confusion_matrix(conf: np.ndarray, class_names: List[str], save_path: str, normalize: bool = True):
    conf_plot = conf.astype(np.float64)
    if normalize:
        row_sums = conf_plot.sum(axis=1, keepdims=True)
        conf_plot = np.divide(conf_plot, np.maximum(row_sums, 1.0))

    fig = plt.figure(figsize=(max(10, len(class_names) * 0.6), max(8, len(class_names) * 0.55)))
    ax = fig.add_subplot(111)
    im = ax.imshow(conf_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Matriz de Confusão" + (" (Normalizada por linha)" if normalize else ""))
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, ha="center", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_ylim(len(class_names) - 0.5, -0.5)  # fix do matplotlib para cortar labels

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def evaluate_model(model, device, val_dir, classes_ckpt, img_size, out_dir):
    tfm = get_val_transform(img_size)
    ds = datasets.ImageFolder(val_dir, transform=tfm)

    # Mapeia índice do dataset -> índice do checkpoint, por nome de classe.
    # Isso garante que métricas e matriz estejam consistentes com o ckpt.
    name_to_ckpt = {name: i for i, name in enumerate(classes_ckpt)}
    idx_ds_to_ckpt = []
    missing = []
    for name in ds.classes:
        if name not in name_to_ckpt:
            missing.append(name)
        idx_ds_to_ckpt.append(name_to_ckpt.get(name, -1))

    if missing:
        raise ValueError(
            "Há classes no dataset de validação que não existem no checkpoint:\n"
            f"{missing}\n"
            "Verifique se está avaliando no mesmo dataset do treino."
        )

    def map_target(y_ds: int) -> int:
        y_ckpt = idx_ds_to_ckpt[y_ds]
        if y_ckpt < 0:
            raise RuntimeError("Falha no mapeamento de classes (idx_ds_to_ckpt).")
        return y_ckpt

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()

    K = len(classes_ckpt)
    conf = np.zeros((K, K), dtype=np.int64)

    total = 0
    correct = 0

    # Para registrar erros mais confiantes
    # item: (prob_pred, path, true_name, pred_name, prob_true, prob_pred)
    errors: List[Tuple[float, str, str, str, float, float]] = []

    with torch.no_grad():
        for batch_idx, (x, y_ds) in enumerate(loader):
            x = x.to(device)

            # mapeia y do dataset -> y do checkpoint
            y_ckpt = torch.tensor([map_target(int(v)) for v in y_ds], dtype=torch.long, device=device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred = probs.argmax(dim=1)

            # Atualiza métricas globais
            total += y_ckpt.numel()
            correct += (pred == y_ckpt).sum().item()

            # Atualiza confusion matrix e coleta erros
            for i in range(y_ckpt.numel()):
                yt = int(y_ckpt[i].item())
                yp = int(pred[i].item())
                conf[yt, yp] += 1

                if yp != yt:
                    # path está em ds.samples (mesma ordem do loader)
                    global_index = batch_idx * loader.batch_size + i
                    img_path = ds.samples[global_index][0]

                    prob_pred = float(probs[i, yp].item())
                    prob_true = float(probs[i, yt].item())
                    errors.append((
                        prob_pred,
                        img_path,
                        classes_ckpt[yt],
                        classes_ckpt[yp],
                        prob_true,
                        prob_pred,
                    ))

    acc = correct / max(1, total)
    macro_p, macro_r, macro_f1, per_p, per_r, per_f1, supports = compute_macro_metrics(conf)

    print("\n===== RESULTADOS (VALIDAÇÃO) =====")
    print(f"Total amostras: {total}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Precision macro: {macro_p:.4f}")
    print(f"Recall macro:    {macro_r:.4f}")
    print(f"F1 macro:        {macro_f1:.4f}")

    # Salva matriz de confusão
    ensure_dir(out_dir)
    cm_path = os.path.join(out_dir, CONFUSION_PNG)
    plot_confusion_matrix(conf, classes_ckpt, cm_path, normalize=True)
    print(f"Matriz de confusão salva em: {cm_path}")

    # Salva top erros
    errors.sort(key=lambda t: t[0], reverse=True)
    top = errors[:min(TOP_ERRORS, len(errors))]

    err_path = os.path.join(out_dir, ERRORS_TXT)
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("Top erros (mais confiantes primeiro)\n")
        f.write("Formato: prob_pred | true -> pred | prob_true | caminho\n\n")
        for prob_pred, img_path, true_name, pred_name, prob_true, prob_pred2 in top:
            f.write(f"{prob_pred2:.4f} | {true_name} -> {pred_name} | {prob_true:.4f} | {img_path}\n")

    print(f"Top erros salvos em: {err_path}")

    # Mostra piores classes (por F1) no console
    order = np.argsort(per_f1)
    print("\nPiores classes por F1 (até 5):")
    for idx in order[:min(5, K)]:
        if supports[idx] == 0:
            continue
        print(f" - {classes_ckpt[idx]} | F1={per_f1[idx]:.3f} P={per_p[idx]:.3f} R={per_r[idx]:.3f} n={supports[idx]}")

    return acc, macro_f1



def predict_single_image(model, device, image_path, img_size, classes, topk=5):
    tfm = get_val_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)

    print(f"\n[IMAGEM] {image_path}")
    print("Top previsões:")
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        print(f" - {classes[idx]}: {p:.4f}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    ckpt = load_checkpoint(CKPT_PATH)
    model, arch, classes, img_size, dropout_p = build_model_from_ckpt(ckpt)
    model = model.to(device)

    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Arquitetura: {arch} | classes: {len(classes)} | img_size: {img_size} | dropout_head: {dropout_p}")

    val_dir = os.path.join(DATA_DIR, VAL_FOLDER)
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Pasta de validação não encontrada: {val_dir}")

    evaluate_model(
        model=model,
        device=device,
        val_dir=val_dir,
        classes_ckpt=classes,
        img_size=img_size,
        out_dir=OUT_DIR,
    )

    if RUN_SINGLE_IMAGE_TEST:
        if not os.path.isfile(IMAGE_PATH):
            raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_PATH}")
        predict_single_image(model, device, IMAGE_PATH, img_size, classes, topk=TOPK_SINGLE)



if __name__ == "__main__":
    main()
