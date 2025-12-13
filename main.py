import os
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
MIN_AREA = 900

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

def seed_signs():
    def add(code, name, cat, desc, pr=0, aliases=None):
        db_upsert_sign(code, name, cat, desc, pr)
        db_upsert_alias(code.lower(), code)
        db_upsert_alias(name.lower(), code)
        if aliases:
            for a in aliases:
                db_upsert_alias(str(a).strip().lower(), code)

    add("B-20", "STOP", "Zakazu", "Zatrzymanie obowiązkowe.", 10, ["stop"])
    add("A-7", "Ustąp pierwszeństwa", "Ostrzegawczy", "Ustąp pierwszeństwa przejazdu.", 9, ["ustap", "yield", "give way"])
    add("D-1", "Droga z pierwszeństwem", "Informacyjny", "Masz pierwszeństwo na tej drodze.", 6, ["priority road"])
    add("D-2", "Koniec drogi z pierwszeństwem", "Informacyjny", "Koniec drogi z pierwszeństwem.", 5, ["end priority road"])

    add("B-33", "Ograniczenie prędkości", "Zakazu", "Nie przekraczaj wskazanej prędkości.", 8, ["speed limit", "limit"])
    add("B-34", "Koniec ograniczenia prędkości", "Zakazu", "Koniec ograniczenia prędkości.", 6, ["end speed limit"])
    add("B-35", "Zakaz postoju", "Zakazu", "Zakaz postoju.", 6, ["no parking"])
    add("B-36", "Zakaz zatrzymywania się", "Zakazu", "Zakaz zatrzymywania się.", 7, ["no stopping"])
    add("B-1", "Zakaz ruchu w obu kierunkach", "Zakazu", "Zakaz ruchu w obu kierunkach.", 8, ["no vehicles"])
    add("B-2", "Zakaz wjazdu", "Zakazu", "Zakaz wjazdu.", 9, ["no entry"])
    add("B-3", "Zakaz wjazdu pojazdów silnikowych", "Zakazu", "Zakaz wjazdu pojazdów silnikowych.", 7, ["no motor vehicles"])
    add("B-9", "Zakaz wjazdu rowerów", "Zakazu", "Zakaz wjazdu rowerów.", 5, ["no bicycles"])
    add("B-10", "Zakaz wjazdu motorowerów", "Zakazu", "Zakaz wjazdu motorowerów.", 4)
    add("B-11", "Zakaz wjazdu wózków rowerowych", "Zakazu", "Zakaz wjazdu wózków rowerowych.", 3)
    add("B-12", "Zakaz wjazdu wózków ręcznych", "Zakazu", "Zakaz wjazdu wózków ręcznych.", 3)
    add("B-13", "Zakaz wjazdu pojazdów z materiałami niebezpiecznymi", "Zakazu", "Zakaz wjazdu pojazdów przewożących materiały niebezpieczne.", 5)
    add("B-14", "Zakaz wjazdu pojazdów z ładunkiem mogącym zanieczyścić wodę", "Zakazu", "Zakaz wjazdu pojazdów z ładunkiem mogącym zanieczyścić wodę.", 3)
    add("B-15", "Zakaz wjazdu pojazdów o szerokości ponad ... m", "Zakazu", "Zakaz wjazdu pojazdów powyżej dopuszczalnej szerokości.", 4, ["max width"])
    add("B-16", "Zakaz wjazdu pojazdów o wysokości ponad ... m", "Zakazu", "Zakaz wjazdu pojazdów powyżej dopuszczalnej wysokości.", 4, ["max height"])
    add("B-17", "Zakaz wjazdu pojazdów o długości ponad ... m", "Zakazu", "Zakaz wjazdu pojazdów powyżej dopuszczalnej długości.", 4, ["max length"])
    add("B-18", "Zakaz wjazdu pojazdów o masie całkowitej ponad ... t", "Zakazu", "Zakaz wjazdu pojazdów powyżej dopuszczalnej masy całkowitej.", 4, ["max weight"])
    add("B-19", "Zakaz wjazdu pojazdów o nacisku osi większym niż ... t", "Zakazu", "Zakaz wjazdu pojazdów o zbyt dużym nacisku osi.", 4, ["axle load"])
    add("B-21", "Zakaz skrętu w lewo", "Zakazu", "Zakaz skrętu w lewo.", 5, ["no left turn"])
    add("B-22", "Zakaz skrętu w prawo", "Zakazu", "Zakaz skrętu w prawo.", 5, ["no right turn"])
    add("B-23", "Zakaz zawracania", "Zakazu", "Zakaz zawracania.", 5, ["no u-turn"])
    add("B-24", "Koniec zakazu wyprzedzania", "Zakazu", "Koniec zakazu wyprzedzania.", 4, ["end no overtaking"])
    add("B-25", "Zakaz wyprzedzania", "Zakazu", "Zakaz wyprzedzania.", 5, ["no overtaking"])
    add("B-26", "Zakaz wyprzedzania przez samochody ciężarowe", "Zakazu", "Zakaz wyprzedzania przez samochody ciężarowe.", 4)
    add("B-27", "Koniec zakazu wyprzedzania przez samochody ciężarowe", "Zakazu", "Koniec zakazu wyprzedzania przez samochody ciężarowe.", 3)
    add("B-28", "Koniec zakazu używania sygnałów dźwiękowych", "Zakazu", "Koniec zakazu używania sygnałów dźwiękowych.", 2)
    add("B-29", "Zakaz używania sygnałów dźwiękowych", "Zakazu", "Zakaz używania sygnałów dźwiękowych.", 3, ["no horn"])
    add("B-31", "Pierwszeństwo nadjeżdżających z przeciwka", "Zakazu", "Masz pierwszeństwo na zwężonym odcinku.", 6)
    add("B-32", "Pierwszeństwo dla nadjeżdżających z przeciwka", "Zakazu", "Ustąp pierwszeństwa na zwężonym odcinku.", 6)

    add("C-1", "Nakaz jazdy w prawo", "Nakazu", "Nakaz jazdy w prawo.", 5)
    add("C-2", "Nakaz jazdy w prawo za znakiem", "Nakazu", "Nakaz ominięcia przeszkody z prawej strony.", 5)
    add("C-3", "Nakaz jazdy w lewo", "Nakazu", "Nakaz jazdy w lewo.", 5)
    add("C-4", "Nakaz jazdy w lewo za znakiem", "Nakazu", "Nakaz ominięcia przeszkody z lewej strony.", 5)
    add("C-5", "Nakaz jazdy prosto", "Nakazu", "Nakaz jazdy prosto.", 5)
    add("C-6", "Nakaz jazdy prosto lub w prawo", "Nakazu", "Nakaz jazdy prosto lub w prawo.", 4)
    add("C-7", "Nakaz jazdy prosto lub w lewo", "Nakazu", "Nakaz jazdy prosto lub w lewo.", 4)
    add("C-8", "Nakaz jazdy w prawo lub w lewo", "Nakazu", "Nakaz jazdy w prawo lub w lewo.", 4)
    add("C-9", "Nakaz jazdy z prawej strony znaku", "Nakazu", "Nakaz omijania znaku z prawej strony.", 3)
    add("C-10", "Nakaz jazdy z lewej strony znaku", "Nakazu", "Nakaz omijania znaku z lewej strony.", 3)
    add("C-11", "Nakaz używania łańcuchów przeciwpoślizgowych", "Nakazu", "Wymagane łańcuchy przeciwpoślizgowe.", 2)
    add("C-12", "Ruch okrężny", "Nakazu", "Obowiązuje ruch okrężny.", 5, ["roundabout"])
    add("C-13", "Droga dla rowerów", "Nakazu", "Droga przeznaczona dla rowerów.", 4, ["bike lane"])
    add("C-13a", "Koniec drogi dla rowerów", "Nakazu", "Koniec drogi dla rowerów.", 3)
    add("C-14", "Prędkość minimalna", "Nakazu", "Nie jedź wolniej niż wskazuje znak.", 3, ["min speed"])
    add("C-15", "Koniec prędkości minimalnej", "Nakazu", "Koniec minimalnej prędkości.", 2)

    add("A-1", "Niebezpieczny zakręt w prawo", "Ostrzegawczy", "Uwaga na niebezpieczny zakręt w prawo.", 3)
    add("A-2", "Niebezpieczny zakręt w lewo", "Ostrzegawczy", "Uwaga na niebezpieczny zakręt w lewo.", 3)
    add("A-3", "Niebezpieczne zakręty", "Ostrzegawczy", "Uwaga na serię niebezpiecznych zakrętów.", 3)
    add("A-4", "Niebezpieczny zjazd", "Ostrzegawczy", "Uwaga na niebezpieczny zjazd.", 2)
    add("A-5", "Niebezpieczne wzniesienie", "Ostrzegawczy", "Uwaga na niebezpieczne wzniesienie.", 2)
    add("A-6a", "Przejazd kolejowy z zaporami", "Ostrzegawczy", "Uwaga na przejazd kolejowy z zaporami.", 5)
    add("A-6b", "Przejazd kolejowy bez zapór", "Ostrzegawczy", "Uwaga na przejazd kolejowy bez zapór.", 5)
    add("A-6c", "Tramwaj", "Ostrzegawczy", "Uwaga na przejazd tramwaju.", 3)
    add("A-7", "Ustąp pierwszeństwa", "Ostrzegawczy", "Ustąp pierwszeństwa przejazdu.", 9)
    add("A-8", "Skrzyżowanie o ruchu okrężnym", "Ostrzegawczy", "Uwaga na rondo.", 3)
    add("A-9", "Przejazd kolejowy bez zapór", "Ostrzegawczy", "Uwaga na przejazd kolejowy.", 4)
    add("A-10", "Nierówna droga", "Ostrzegawczy", "Uwaga na nierówności.", 2)
    add("A-11", "Próg zwalniający", "Ostrzegawczy", "Uwaga na próg zwalniający.", 3, ["speed bump"])
    add("A-12a", "Zwężenie jezdni", "Ostrzegawczy", "Uwaga na zwężenie jezdni.", 3)
    add("A-12b", "Zwężenie jezdni - prawostronne", "Ostrzegawczy", "Uwaga na zwężenie z prawej strony.", 3)
    add("A-12c", "Zwężenie jezdni - lewostronne", "Ostrzegawczy", "Uwaga na zwężenie z lewej strony.", 3)
    add("A-13", "Ruch dwukierunkowy", "Ostrzegawczy", "Uwaga, ruch dwukierunkowy.", 3)
    add("A-14", "Roboty na drodze", "Ostrzegawczy", "Uwaga, roboty drogowe.", 3, ["road works"])
    add("A-15", "Śliska jezdnia", "Ostrzegawczy", "Uwaga, śliska jezdnia.", 3, ["slippery road"])
    add("A-16", "Przejście dla pieszych", "Ostrzegawczy", "Uwaga, przejście dla pieszych.", 5, ["pedestrian crossing"])
    add("A-17", "Dzieci", "Ostrzegawczy", "Uwaga, dzieci.", 4, ["children"])
    add("A-18a", "Zwierzęta gospodarskie", "Ostrzegawczy", "Uwaga, zwierzęta gospodarskie.", 2)
    add("A-18b", "Dzikie zwierzęta", "Ostrzegawczy", "Uwaga, dzikie zwierzęta.", 2)
    add("A-19", "Boczny wiatr", "Ostrzegawczy", "Uwaga na boczny wiatr.", 2)
    add("A-20", "Odcinek jezdni o ruchu wahadłowym", "Ostrzegawczy", "Możliwy ruch wahadłowy.", 2)
    add("A-21", "Tramwaj", "Ostrzegawczy", "Uwaga na tramwaj.", 2)
    add("A-22", "Niebezpieczny spadek", "Ostrzegawczy", "Uwaga na spadek.", 2)

    add("D-6", "Przejście dla pieszych", "Informacyjny", "Przejście dla pieszych.", 5, ["crosswalk"])
    add("D-6a", "Przejazd dla rowerzystów", "Informacyjny", "Przejazd dla rowerzystów.", 4, ["bike crossing"])
    add("D-6b", "Przejście dla pieszych i przejazd dla rowerzystów", "Informacyjny", "Przejście i przejazd rowerowy.", 4)
    add("D-7", "Droga ekspresowa", "Informacyjny", "Wjazd na drogę ekspresową.", 3)
    add("D-8", "Koniec drogi ekspresowej", "Informacyjny", "Koniec drogi ekspresowej.", 2)
    add("D-9", "Autostrada", "Informacyjny", "Wjazd na autostradę.", 3)
    add("D-10", "Koniec autostrady", "Informacyjny", "Koniec autostrady.", 2)
    add("D-11", "Początek pasa ruchu dla autobusów", "Informacyjny", "Początek buspasa.", 2, ["bus lane"])
    add("D-12", "Koniec pasa ruchu dla autobusów", "Informacyjny", "Koniec buspasa.", 2)
    add("D-18", "Parking", "Informacyjny", "Parking.", 2, ["parking"])
    add("D-18a", "Parking zadaszony", "Informacyjny", "Parking zadaszony.", 1)
    add("D-23", "Stacja paliw", "Informacyjny", "Stacja paliw.", 1, ["gas station", "fuel"])
    add("D-24", "Telefon", "Informacyjny", "Telefon.", 1)
    add("D-25", "Punkt informacji turystycznej", "Informacyjny", "Informacja turystyczna.", 1)
    add("D-26", "Punkt pierwszej pomocy", "Informacyjny", "Punkt pierwszej pomocy.", 2, ["first aid"])
    add("D-34", "Punkt kontroli drogowej", "Informacyjny", "Możliwa kontrola drogowa.", 1)

def db_lookup(label):
    if not label:
        return None
    key = str(label).strip().lower()
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

def put_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    px = max(0, x)
    py = max(0, y - 8)
    cv2.rectangle(img, (px, py - th - 10), (px + tw + 12, py + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (px + 6, py - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

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

    return red, blue, yellow

def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return None
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    corners = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    ar = (w / h) if h else 0
    if corners == 8:
        return ("stop", (x, y, x + w, y + h), 0.55)
    if corners == 3:
        return ("a-7", (x, y, x + w, y + h), 0.45)
    if corners >= 9 and 0.75 <= ar <= 1.25:
        return ("speed limit", (x, y, x + w, y + h), 0.35)
    if corners == 4 and 0.75 <= ar <= 1.35:
        return ("informacyjny", (x, y, x + w, y + h), 0.25)
    return None

def detect_fallback(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red, blue, yellow = hsv_masks(hsv)
    kernel = np.ones((5, 5), np.uint8)
    out = []
    for m in (red, blue, yellow):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            r = classify_shape(cnt)
            if not r:
                continue
            label, bbox, conf = r
            out.append({"label": label, "conf": float(conf), "bbox": bbox})
    return out

def load_model_or_none():
    if not YOLO_AVAILABLE:
        return None
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH)
        except Exception:
            return None
    return None

def detect_yolo(model, img):
    preds = model.predict(img, conf=CONF_TH, iou=IOU_TH, verbose=False)
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
        r.append({"label": str(label).strip(), "conf": float(c), "bbox": (int(x1), int(y1), int(x2), int(y2))})
    return r

def enrich(dets):
    enriched = []
    for d in dets:
        info = db_lookup(d["label"])
        if info is None:
            info = {"code": d["label"], "name": d["label"], "category": None, "desc": None, "priority": 0}
        enriched.append({**d, **info})
    enriched.sort(key=lambda x: (x.get("priority", 0), x.get("conf", 0.0)), reverse=True)
    return enriched

def draw(img, enriched):
    out = img.copy()
    for d in enriched:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d["name"]} ({d["conf"]:.2f})'
        put_label(out, label, x1, y1)
    return out

def main():
    db_init()
    seed_signs()
    engine = speak_init()
    model = load_model_or_none()

    image_path = "znak.png"
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"BŁĄD: Nie widzę pliku '{image_path}'.")
        return

    if model is not None:
        dets = detect_yolo(model, img)
    else:
        dets = detect_fallback(img)

    enriched = enrich(dets)
    out = draw(img, enriched)

    if enriched:
        print("\n" + "#" * 40)
        for d in enriched[:12]:
            print(f'WYKRYTO: {d["name"]} | pewność: {d["conf"]:.2f} | kod: {d["code"]}')
            if d.get("desc"):
                print(f'OPIS: {d["desc"]}')
        print("#" * 40 + "\n")
        top = enriched[0]
        msg = top.get("desc") or top.get("name") or ""
        if msg:
            speak(engine, msg)
    else:
        print("Nie wykryto znaku.")

    cv2.imshow("SignAI", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
