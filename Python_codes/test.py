import cv2
from Sign_detect import SignDetector   # <-- upewnij się, że plik nazywa się sign_detector.py

def main():
    # 1. Inicjalizacja detektora
    detector = SignDetector(
        debug=False,          # ustaw True jeśli chcesz logi w konsoli
        display=False,         # pokazuje wynikowy obraz
        base_path="sign_database",
        preload=True          # od razu ładuje całą bazę
    )

    # 2. Inicjalizacja kamery (0 = domyślna kamera laptopa)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera nie jest dostępna.")
        return

    print("[INFO] Uruchomiono kamerę. Naciśnij 'q' aby zakończyć.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Nie udało się pobrać klatki.")
            break

        # Opcjonalnie zmniejsz rozdzielczość (dla wydajności)
        frame = cv2.resize(frame, (640, 480))

        # 3. Wykrywanie znaków
        result = detector.detect(frame)

        # 4. Wyświetl liczbę wykryć i ich nazwy w konsoli
        detections = result["detections"]
        if result["detected"]:
            print(f"[INFO] Wykryto {len(detections)} znak(ów):")
            for d in detections:
                print(f"   - {d['name']} ({d['score']:.1f}%) [{d['color']} {d['shape']}]")

        # 5. Wyświetlanie obrazu (robione już w klasie, jeśli display=True)
        # ale na wszelki wypadek zostawiam:
        cv2.imshow("Detected Output", result["output_frame"])

        # 6. Wyjście po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Zakończono test.")
            break

    # 7. Zakończenie
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
