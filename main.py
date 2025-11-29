import cv2
import numpy as np
import pyttsx3

# --- 1. KONFIGURACJA GŁOSU (ZMIANA NA MĘSKI) ---
def speak(text):
    try:
        engine = pyttsx3.init()
        
        # Pobranie listy dostępnych głosów w systemie
        voices = engine.getProperty('voices')
        
        # Ustawienie głosu MĘSKIEGO. 
        # Zazwyczaj voices[0] to mężczyzna (np. Adam/David), a voices[1] to kobieta.
        # Jeśli ten głos będzie żeński, zmień [0] na [1].
        if len(voices) > 0:
            engine.setProperty('voice', voices[0].id) 
            
        engine.setProperty('rate', 150)  # Szybkość mowy
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Błąd modułu mowy: {e}")

# --- 2. LOGIKA ROZPOZNAWANIA KSZTAŁTÓW I KOLORÓW ---
def identify_sign(image):
    output_img = image.copy()
    
    # Krok A: Konwersja do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Krok B: Definicja koloru CZERWONEGO
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask1 + mask2
    
    # Krok C: Znajdowanie konturów
    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_message = ""
    detected_name = ""

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: 
            continue
            
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        corners = len(approx)
        
        x, y, w, h = cv2.boundingRect(approx)
        
        # --- LOGIKA DECYZYJNA ---
        if corners == 8: # STOP
            detected_name = "STOP"
            detected_message = "Znak STOP. Zatrzymaj się bezwzględnie."
            cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 3)

        elif corners == 3: # TRÓJKĄT
            detected_name = "USTAP PIERWSZENSTWA"
            detected_message = "Znak Ustąp pierwszeństwa przejazdu."
            cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 3)

        elif corners > 8: # KOŁO
            aspect_ratio = float(w)/h
            if 0.8 <= aspect_ratio <= 1.2:
                detected_name = "ZAKAZ / OGRANICZENIE"
                detected_message = "Znak Zakazu lub Ograniczenia prędkości."
                cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 3)

    return output_img, detected_name, detected_message

# --- 3. GŁÓWNA PĘTLA PROGRAMU ---
def main():
    image_path = "znak.jpg" 
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"BŁĄD: Nie widzę pliku '{image_path}'")
        return

    # --- SKALOWANIE (Żeby okno nie było za duże) ---
    max_width = 800
    if img.shape[1] > max_width:
        scale_ratio = max_width / img.shape[1]
        new_width = int(img.shape[1] * scale_ratio)
        new_height = int(img.shape[0] * scale_ratio)
        img = cv2.resize(img, (new_width, new_height))
    # -----------------------------------------------

    print("Przetwarzanie obrazu...")
    result_img, name, message = identify_sign(img)
    
    if name:
        # A. Wizualizacja
        text_on_screen = f"{name}"
        cv2.putText(result_img, text_on_screen, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # B. Konsola
        print("\n" + "#"*30)
        print(f"ROZPOZNANO: {name}")
        print(f"LEKTOR: {message}")
        print("#"*30 + "\n")
        
        # C. Okno
        cv2.imshow("SignAI Detection", result_img)
        
        # D. Głos
        speak(message)
        
        cv2.waitKey(0) 
    else:
        print("Nie wykryto znaku.")
        cv2.imshow("Original", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()