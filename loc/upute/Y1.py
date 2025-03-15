from ultralytics import YOLO

def main():
    # 1) Učitati istrenirani model
    model = YOLO("runs/detect/train2/weights/best.pt")  
    #    - Ovdje stavite točan path do svojih treniranih težina

    # 2) Pokrenuti inference na nekom setu slika (npr. test images)
    #    i reći YOLO-u da snimi:
    #       - slike s bounding boxovima (save=True)
    #       - tekstualne .txt fajlove s koordinatama (save_txt=True)
    results = model.predict(
        source="/home/team11/dev/MediSense/loc/upute/processed_dataset/test/images",  # mapu slika za test (ili val)
        conf=0.15,        # prag za confidence, prilagodite po želji
        save=True,        # spremi output slike
        save_txt=True,    # generiraj .txt s bounding boxovima
        project="runs/detect",  # glavni output direktorij
        name="predict_test"     # ime pod-mape
    )

    # Nakon ovoga dobit ćete .txt fajlove u:
    # runs/detect/predict_test/labels/
    # (npr. 123_45.txt za svaku sliku),
    # koje onda čita post_processing.py

if __name__ == "__main__":
    main()