import os
import time
import sqlite3
import cv2
import numpy as np
import pyttsx3

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

DB_PATH = "signai_signs.db"
MODEL_PATH = "traffic_sign_yolo.pt"
CONF_TH = 0.35
IOU_TH = 0.45
SPEAK_RATE = 155
SPEAK_COOLDOWN_SEC = 2.0
MAX_SPEAK_PER_FRAME = 2

def speak_init():
    engine = pyttsx3.init()
    engine.setProperty("rate", SPEAK_RATE)
    return engine

def speak(engine, text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def db_connect():
    return sqlite3.connect(DB_PATH)

def db_init():
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE,
            name_pl TEXT NOT NULL,
            category TEXT,
            description_pl TEXT,
            priority INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT UNIQUE,
            code TEXT NOT NULL,
            FOREIGN KEY(code) REFERENCES signs(code)
        )
    """)
    con.commit()
    con.close()

def db_upsert_sign(code, name_pl, category, description_pl, priority=0):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO signs(code, name_pl, category, description_pl, priority)
        VALUES(?,?,?,?,?)
        ON CONFLICT(code) DO UPDATE SET
            name_pl=excluded.name_pl,
            category=excluded.category,
            description_pl=excluded.description_pl,
            priority=excluded.priority
    """, (code, name_pl, category, description_pl, int(priority)))
    con.commit()
    con.close()

def db_upsert_alias(alias, code):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO aliases(alias, code)
        VALUES(?,?)
        ON CONFLICT(alias) DO UPDATE SET
            code=excluded.code
    """, (alias, code))
    con.commit()
    con.close()

def db_seed_minimal():
    db_upsert_sign("STOP", "STOP", "Zakaz / Obowiązek zatrzymania", "Znak STOP. Zatrzymaj się bezwzględnie.", 10)
    db_upsert_sign("A-7", "Ustąp pierwszeństwa", "Ostrzegawczy", "Ustąp pierwszeństwa przejazdu.", 9)
    db_upsert_sign("B-20", "STOP", "Zakazu", "Zatrzymanie obowiązkowe.", 10)
    db_upsert_sign("B-33", "Ograniczenie prędkości", "Zakazu", "Nie przekraczaj wskazanej prędkości.", 7)
    db_upsert_sign("B-36", "Zakaz zatrzymywania się", "Zakazu", "Zakaz zatrzymywania się.", 6)
    db_upsert_sign("B-35", "Zakaz postoju", "Zakazu", "Zakaz postoju.", 6)
    db_upsert_sign("C-2", "Nakaz jazdy w prawo", "Nakazu", "Nakaz jazdy w prawo za znakiem.", 5)
    db_upsert_sign("C-4", "Nakaz jazdy w lewo", "Nakazu", "Nakaz jazdy w lewo za znakiem.", 5)
    db_upsert_sign("D-1", "Droga z pierwszeństwem", "Informacyjny", "Masz pierwszeństwo na tej drodze.", 5)
    db_upsert_sign("D-2", "Koniec drogi z pierwszeństwem", "Informacyjny", "Koniec drogi z pierwszeństwem.", 4)
    db_upsert_sign("A-1", "Niebezpieczny zakręt w prawo", "Ostrzegawczy", "Uwaga na niebezpieczny zakręt w prawo.", 4)
    db_upsert_sign("A-2", "Niebezpieczny zakręt w lewo", "Ostrzegawczy", "Uwaga na niebezpieczny zakręt w lewo.", 4)
    db_upsert_sign("A-3", "Niebezpieczne zakręty", "Ostrzegawczy", "Uwaga na serię niebezpiecznych zakrętów.", 4)
    db_upsert_sign("A-6a", "Przejazd kolejowy z zaporami", "Ostrzegawczy", "Uwaga na przejazd kolejowy z zaporami.", 6)
    db_upsert_sign("A-6b", "Przejazd kolejowy bez zapór", "Ostrzegawczy", "Uwaga na przejazd kolejowy bez zapór.", 6)
    db_upsert_sign("A-16", "Przejście dla pieszych", "Ostrzegawczy", "Uwaga, przejście dla pieszych.", 5)
    db_upsert_sign("D-6", "Przejście dla pieszych", "Informacyjny", "Przejście dla pieszych.", 5)
    db_upsert_sign("D-6a", "Przejazd dla rowerzystów", "Informacyjny", "Przejazd dla rowerzystów.", 4)
    db_upsert_sign("C-13", "Droga dla rowerów", "Nakazu", "Droga przeznaczona dla rowerów.", 3)
    db_upsert_sign("B-2", "Zakaz wjazdu", "Zakazu", "Zakaz wjazdu.", 8)

    db_upsert_alias("stop", "STOP")
    db_upsert_alias("a-7", "A-7")
    db_upsert_alias("yield", "A-7")
    db_upsert_alias("speed limit", "B-33")
    db_upsert_alias("speed_limit", "B-33")
    db_upsert_alias("no entry", "B-2")
    db_upsert_alias("no_entry", "B-2")
    db_upsert_alias("crosswalk", "D-6")
    db_upsert_alias("pedestrian crossing", "D-6")
    db_upsert_alias("priority road", "D-1")
    db_upsert_alias("end of priority road", "D-2")

def db_lookup(label):
    if not label:
        return None
    key = label.strip().lower()
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT s.code, s.name_pl, s.category, s.description_pl, s.priority
        FROM aliases a
        JOIN signs s ON s.code = a.code
        WHERE a.alias = ?
        LIMIT 1
    """, (key,))
    row = cur.fetchone()
    if row is None:
        cur.execute("""
            SELECT code, name_pl, category, description_pl, priority
            FROM signs
            WHERE lower(code)=? OR lower(name_pl)=?
            LIMIT 1
        """, (key, key))
        row = cur.fetchone()
    con.close()
    if row is None:
        return None
    return {"code": row[0], "name": row[1], "category": row[2], "desc": row[3], "priority": int(row[4] or 0)}

def norm_label(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("—", "-").replace("–", "-").replace("_", " ")
    return " ".join(s.split())

def put_label(img, text, x, y, w, h):
    text = norm_label(text)
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    px = max(0, x)
    py = max(0, y - 10)
    cv2.rectangle(img, (px, py - th - 8), (px + tw + 10, py + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (px + 5, py - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def hsv_masks(hsv):
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    white = cv2.inRange(hsv, lower_white, upper_white)

    return red, blue, yellow, white

def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < 1200:
        return None
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    epsilon = 0.03 * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    corners = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    ar = (w / h) if h else 0
    if corners == 8:
        return {"label": "stop", "bbox": (x, y, w, h), "approx": approx, "score": 0.55}
    if corners == 3:
        return {"label": "a-7", "bbox": (x, y, w, h), "approx": approx, "score": 0.45}
    if corners >= 9 and 0.75 <= ar <= 1.25:
        return {"label": "speed limit", "bbox": (x, y, w, h), "approx": approx, "score": 0.35}
    if corners == 4 and 0.75 <= ar <= 1.35:
        return {"label": "informacyjny", "bbox": (x, y, w, h), "approx": approx, "score": 0.25}
    return None

def detect_fallback(frame):
    out = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red, blue, yellow, white = hsv_masks(hsv)
    kernel = np.ones((5, 5), np.uint8)
    masks = [("red", red), ("blue", blue), ("yellow", yellow)]
    for name, m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            r = classify_shape(cnt)
            if not r:
                continue
            x, y, w, h = r["bbox"]
            if w < 25 or h < 25:
                continue
            out.append({"label": r["label"], "conf": float(r["score"]), "bbox": (x, y, x + w, y + h)})
    return out

def detect_yolo(model, frame):
    preds = model.predict(frame, conf=CONF_TH, iou=IOU_TH, verbose=False)
    r = []
    if not preds:
        return r
    p = preds[0]
    names = getattr(p, "names", None) or getattr(model, "names", None)
    boxes = getattr(p, "boxes", None)
    if boxes is None:
        return r
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        label = str(k)
        if isinstance(names, dict) and k in names:
            label = names[k]
        elif isinstance(names, (list, tuple)) and 0 <= k < len(names):
            label = names[k]
        r.append({"label": norm_label(label), "conf": float(c), "bbox": (int(x1), int(y1), int(x2), int(y2))})
    return r

def best_messages(detections):
    enriched = []
    for d in detections:
        info = db_lookup(d["label"])
        if info is None:
            info = {"code": d["label"], "name": d["label"], "category": None, "desc": None, "priority": 0}
        enriched.append({**d, **info})
    enriched.sort(key=lambda x: (x.get("priority", 0), x.get("conf", 0.0)), reverse=True)
    return enriched

def draw_detections(frame, enriched):
    out = frame.copy()
    for d in enriched:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d["name"]} ({d["conf"]:.2f})'
        put_label(out, label, x1, y1, x2 - x1, y2 - y1)
    return out

def should_speak(now, last_time):
    return (now - last_time) >= SPEAK_COOLDOWN_SEC

def build_speech_queue(enriched):
    q = []
    used = set()
    for d in enriched:
        msg = d.get("desc") or d.get("name") or ""
        msg = norm_label(msg)
        if not msg:
            continue
        k = msg.lower()
        if k in used:
            continue
        used.add(k)
        q.append(msg)
        if len(q) >= MAX_SPEAK_PER_FRAME:
            break
    return q

def run_on_image(path, model, engine):
    img = cv2.imread(path)
    if img is None:
        print(f"BŁĄD: Nie widzę pliku '{path}'.")
        return
    if model is not None:
        det = detect_yolo(model, img)
    else:
        det = detect_fallback(img)
    enriched = best_messages(det)
    out = draw_detections(img, enriched)
    if enriched:
        print("\n" + "#" * 40)
        for d in enriched[:8]:
            print(f'WYKRYTO: {d["name"]} | pewność: {d["conf"]:.2f} | kod: {d["code"]}')
            if d.get("desc"):
                print(f'OPIS: {d["desc"]}')
        print("#" * 40 + "\n")
        queue = build_speech_queue(enriched)
        for t in queue:
            speak(engine, t)
    else:
        print("Nie wykryto znaku.")
    cv2.imshow("SignAI", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(source, model, engine):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("BŁĄD: Nie mogę otworzyć źródła wideo.")
        return
    last_spoken = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if model is not None:
            det = detect_yolo(model, frame)
        else:
            det = detect_fallback(frame)
        enriched = best_messages(det)
        out = draw_detections(frame, enriched)
        now = time.time()
        if enriched and should_speak(now, last_spoken):
            queue = build_speech_queue(enriched)
            if queue:
                speak(engine, queue[0])
                last_spoken = now
        cv2.imshow("SignAI", out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def load_model_or_none():
    if not YOLO_AVAILABLE:
        return None
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH)
        except Exception:
            return None
    return None

def main():
    db_init()
    db_seed_minimal()
    engine = speak_init()
    model = load_model_or_none()

    print("SignAI uruchomione.")
    print("Tryby:")
    print("1) obraz:  python signai.py image znak.jpg")
    print("2) kamera: python signai.py cam")
    print("3) wideo:  python signai.py video film.mp4")
    print(f"Model YOLO: {'TAK' if model is not None else 'NIE (fallback OpenCV)'}")
    import sys
    args = sys.argv[1:]
    if not args:
        run_on_image("znak.jpg", model, engine)
        return
    mode = args[0].lower()
    if mode == "image":
        path = args[1] if len(args) > 1 else "znak.jpg"
        run_on_image(path, model, engine)
        return
    if mode == "cam":
        run_on_video(0, model, engine)
        return
    if mode == "video":
        path = args[1] if len(args) > 1 else "film.mp4"
        run_on_video(path, model, engine)
        return
    run_on_image(args[0], model, engine)

if __name__ == "__main__":
    main()
