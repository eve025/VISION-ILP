import cv2
import numpy as np

# 1. DEFINICIÓN DE VARIABLES
# ==========================================
# Caso 9
img09_entrada, img09_salida = 'portabaterias.png', 'etapas_img09.png'
# Caso 10 (A y B)
img10a_entrada, img10a_salida = 'pinza-mesa-120deg.png', 'etapas_img10_120.png'
img10b_entrada, img10b_salida = 'pinza-mesa-90deg.png', 'etapas_img10_90.png'
# Caso 11
img11_entrada, img11_salida = 'Perno-de-bloqueo.png', 'etapas_img11.png'
# Caso 12
img12_entrada, img12_salida = 'placa-rectangular.png', 'etapas_img12.png'

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO 
# ==========================================
def procesar_figura(nombre_imagen, nombre_salida):
    # 1. Cargar imagen
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_img = img.copy()

    # 2. Detección de bordes
    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Procesa cada contorno
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"({cx},{cy})", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)

            length = 40
            angle_rad = np.deg2rad(angle)
            dx = int(length * np.cos(angle_rad))
            dy = int(length * np.sin(angle_rad))
            cv2.line(color_img, (cx, cy), (cx + dx, cy + dy), (0, 255, 0), 2)

            print(f"Imagen: {nombre_imagen} -> Centro: ({cx}, {cy}), Angulo: {angle} deg")

    # --- CREAR MOSAICO (Original, bordes , resultado) ---
    h, w = img.shape[:2]
    nueva_w = 400
    nueva_h = int(h * (nueva_w / w))
    
    res1 = cv2.resize(img, (nueva_w, nueva_h))
    res2 = cv2.resize(edges_bgr, (nueva_w, nueva_h))
    res3 = cv2.resize(color_img, (nueva_w, nueva_h))
    
    mosaico = np.hstack((res1, res2, res3))

    # 5. Muestra y guarda
    cv2.imshow(f'Procesando: {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) # Pausa para ver cada una
    cv2.destroyAllWindows()

# =====================================================
# 3. EJECUCIÓN 
procesar_figura(img09_entrada, img09_salida)
procesar_figura(img10a_entrada, img10a_salida)
procesar_figura(img10b_entrada, img10b_salida)
procesar_figura(img11_entrada, img11_salida)
procesar_figura(img12_entrada, img12_salida)

print("Proceso finalizado.")