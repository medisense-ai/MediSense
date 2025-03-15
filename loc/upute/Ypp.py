import os
import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Isti poredak klasa koji ste imali i pri treniranju (YOLO class IDs)
id2category = {
    0: "Mass",
    1: "Suspicious_Calcification",
    2: "Focal_Asymmetry",
    3: "Architectural_Distortion",
    4: "Suspicious_Lymph_Node",
    5: "Other",
}

def compute_letterbox_params(orig_w, orig_h, new_w=640, new_h=640):
    """
    Računa faktore (r, dw, dh) jednako kao u 'letterbox_image' fazi.
    r     = faktor skaliranja
    dw, dh = offseti u 640x640 (broj piksela 'paddinga' slijeva i s gornje strane).
    """
    r = min(new_w / orig_w, new_h / orig_h)
    # new unpadded size nakon skaliranja
    resized_w, resized_h = int(orig_w * r), int(orig_h * r)
    dw = (new_w - resized_w) // 2
    dh = (new_h - resized_h) // 2
    return r, dw, dh

def unletterbox_yolo_coords(x_center_n, y_center_n, w_n, h_n, 
                            img_w_new, img_h_new, 
                            dw, dh, r, 
                            orig_w, orig_h):
    """
    'Undo' letterbox i normalizirane YOLO koordinate (640x640) 
    natrag u originalne dimenzije slike (orig_w x orig_h).

    (x_center_n, y_center_n, w_n, h_n) - normalizirani [0..1] iz YOLO .txt
    (img_w_new, img_h_new)            - veličina (640,640) nakon letterboxa
    (dw, dh)                          - offseti unutar 640x640
    r                                  - scale factor
    (orig_w, orig_h)                  - originalne dimenzije slike

    Vraća (xmin, ymin, xmax, ymax) u koordinatnom sustavu originalne slike.
    """
    # 1) iz [0..1] u [0..640]
    x_center_abs = x_center_n * img_w_new
    y_center_abs = y_center_n * img_h_new
    w_abs        = w_n * img_w_new
    h_abs        = h_n * img_h_new

    # 2) koordinate unutar letterbox slike 640x640
    xmin_lb = x_center_abs - w_abs / 2.0
    xmax_lb = x_center_abs + w_abs / 2.0
    ymin_lb = y_center_abs - h_abs / 2.0
    ymax_lb = y_center_abs + h_abs / 2.0

    # 3) makni offset (dw, dh), pa podijeli s r
    xmin_un = (xmin_lb - dw) / r
    xmax_un = (xmax_lb - dw) / r
    ymin_un = (ymin_lb - dh) / r
    ymax_un = (ymax_lb - dh) / r

    # 4) clamp na [0, orig_dim-1]
    xmin_un = max(0, min(xmin_un, orig_w - 1))
    xmax_un = max(0, min(xmax_un, orig_w - 1))
    ymin_un = max(0, min(ymin_un, orig_h - 1))
    ymax_un = max(0, min(ymax_un, orig_h - 1))

    return int(xmin_un), int(ymin_un), int(xmax_un), int(ymax_un)

def main():
    """
    Glavna post-processing rutina:
    1) Učitava YOLO txt predikcije (class, x_center, y_center, w, h, [confidence])
    2) 'Unletterboxira' bbox-ove u originalne dimenzije
    3) Generira binarne maske (po klasi)
    4) Kreira 'classification_results.csv' (multi-label info).
    """

    # --------------------------------------
    # 1) Definirajte putanje
    # --------------------------------------
    # Gdje YOLO sprema txt datoteke, npr. 'runs/detect/predict/labels'
    yolo_preds_dir = Path("/home/team11/dev/MediSense/loc/upute/runs/detect/predict_test/labels")

    # Originalne slike (bez letterboxa), organizirane npr. <images_base_dir>/<case_id>/<image_id>.jpg
    orig_images_dir = Path("/home/team11/data/train/images")

    # Gdje ćemo spremiti maske za svaku klasu: localization_results/<case_id>/<image_id>/<klasa>.png
    output_localization_dir = Path("localization_results")
    output_localization_dir.mkdir(exist_ok=True, parents=True)

    # --------------------------------------
    # 2) Za multi-label classification CSV
    # --------------------------------------
    categories = list(id2category.values())  # ["Mass", "Suspicious_Calcification", ...]
    classification_rows = []

    # --------------------------------------
    # 3) Prođi kroz sve YOLO txt predikcije
    #    Pretpostavka: <caseID>_<imageID>.txt
    # --------------------------------------
    all_txt_files = sorted(yolo_preds_dir.glob("*.txt"))
    for txt_path in all_txt_files:
        base_name = txt_path.stem  # npr. "123_456"
        parts = base_name.split("_")
        if len(parts) < 2:
            print(f"Ne mogu parsirati case_id i image_id iz '{base_name}' (očekujem npr. 123_456). Preskačem.")
            continue
        case_id = parts[0]
        image_id = "_".join(parts[1:])  # ako postoji više underscorea

        # Put do originalne slike
        img_path = orig_images_dir / case_id / f"{image_id}.jpg"
        if not img_path.exists():
            print(f"[Upozorenje] Originalna slika ne postoji: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Upozorenje] Ne mogu učitati {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]

        # Kreiramo prazan mask array za svaku klasu (originalna rezolucija!)
        masks_per_class = {
            cat_name: np.zeros((orig_h, orig_w), dtype=np.uint8) 
            for cat_name in categories
        }

        # Klasifikacijski indicator (0/1) je li klasa detektirana
        class_flags = {cat_name: 0 for cat_name in categories}

        # Učitaj YOLO predikcije iz .txt
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # Izračunaj (r, dw, dh) identično letterboxu
        # 640x640 jer je target_size=(640,640) u train skripti
        r, dw, dh = compute_letterbox_params(orig_w, orig_h, 640, 640)

        # Parsiraj svaku liniju .txt -> class_id, x_center, y_center, w, h (+/- conf)
        for line in lines:
            vals = line.strip().split()
            if len(vals) < 5:
                continue
            cls_id = int(vals[0])
            x_center_n = float(vals[1])
            y_center_n = float(vals[2])
            w_n        = float(vals[3])
            h_n        = float(vals[4])
            # Ako postoji confidence, to je vals[5] (možete spremiti ako želite)

            # Dohvati ime klase iz ID-a
            cat_name = id2category.get(cls_id, None)
            if cat_name is None:
                continue

            # Označi da je ta klasa barem jednom detektirana
            class_flags[cat_name] = 1

            # Unletterbox
            xmin, ymin, xmax, ymax = unletterbox_yolo_coords(
                x_center_n, y_center_n, w_n, h_n,
                640, 640,
                dw, dh, r,
                orig_w, orig_h
            )

            # Popuni masku: bounding box od (ymin, xmin) do (ymax, xmax) = 255
            masks_per_class[cat_name][ymin:ymax+1, xmin:xmax+1] = 255

        # --------------------------------------
        # 4) Spremi maske u: localization_results/<case_id>/<image_id>/klasa.png
        # --------------------------------------
        out_case_dir = output_localization_dir / case_id / image_id
        out_case_dir.mkdir(exist_ok=True, parents=True)

        for cat_name, mask_arr in masks_per_class.items():
            out_mask_path = out_case_dir / f"{cat_name}.png"
            cv2.imwrite(str(out_mask_path), mask_arr)

        # --------------------------------------
        # 5) Multi-label classification
        # --------------------------------------
        row = {
            "case_id": case_id,
            "image_id": image_id
        }
        for cat in categories:
            row[cat] = class_flags[cat]

        classification_rows.append(row)

    # --------------------------------------
    # 6) Izradi 'classification_results.csv'
    # --------------------------------------
    df = pd.DataFrame(classification_rows)
    df = df[["case_id", "image_id"] + categories]  # Reorder kolona radi čitljivosti
    out_csv = "localization_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Kreiran '{out_csv}' s {len(df)} redaka.")
    print(f"[INFO] Binarne maske spremljene u '{output_localization_dir}'.")


if __name__ == "__main__":
    main()