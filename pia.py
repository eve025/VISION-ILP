import cv2
import numpy as np

# 1. DEFINICIÓN DE VARIABLES
# ==========================================
img09_entrada, img09_salida = 'portabaterias.png', 'etapas_img09.png'
img10a_entrada, img10a_salida = 'pinza-mesa-120deg.png', 'etapas_img10_120.png'
img10b_entrada, img10b_salida = 'pinza-mesa-90deg.png', 'etapas_img10_90.png'
img11_entrada, img11_salida = 'Perno-de-bloqueo.png', 'etapas_img11.png'
img12_entrada, img12_salida = 'placa-rectangular.png', 'etapas_img12.png'

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO 
# ==========================================
def procesar_figura(nombre_imagen, nombre_salida, K, OFF_X, OFF_Y, x_real, y_real, ang_real, es_perno=False):
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    color_img = img.copy()

    # Bordes (Canny)
    low, high = (5, 40) if es_perno else (20, 100)
    edges = cv2.Canny(blurred, low, high)

    # --- Kernel de 5x5 para mantener la forma real de la pinza ---
    kernel = np.ones((5, 5), np.uint8) 
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # Centroide en píxeles
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            # Conversión a coordenadas del Robot
            cx_robot = (cx * K) + OFF_X
            cy_robot = OFF_Y - (cy * K)

            # Cálculo de Orientación
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            angle_robot = angle + 180 if rect[1][0] < rect[1][1] else angle + 90

            # Normalización de ángulo
            while abs(angle_robot - ang_real) > 45:
                if angle_robot > ang_real: angle_robot -= 90
                else: angle_robot += 90

            # --- CÁLCULO DE ERROR Y PRECISIÓN ---
            err_dist = np.sqrt((cx_robot - x_real)**2 + (cy_robot - y_real)**2)
            
            # Precisión relativa (Suponiendo rango de 2000mm)
            precision = (1 - (err_dist / 2000)) * 100 

            # --- DIBUJAR ---
            cv2.circle(color_img, (cx, cy), 4, (0, 0, 255), -1)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Texto 1: Posición (Capa Verde)
            cv2.putText(color_img, f"({int(cx_robot)},{int(cy_robot)})", (cx+10, cy-10), 
                        font, 0.4, (0, 255, 0), 1)
            
            # Texto 2: Ángulo (Capa Amarilla)
            cv2.putText(color_img, f"{angle_robot:.1f}deg", (cx+10, cy+10), 
                        font, 0.4, (0, 255, 255), 1)

            # --- CONSOLA DETALLADA ---
            print(f"FICHA: {nombre_imagen}")
            print(f"  > Píxeles: ({cx}, {cy})")
            print(f"  > Robot Calc: X={cx_robot:.2f}, Y={cy_robot:.2f}")
            print(f"  > Robot Real: X={x_real}, Y={y_real}")
            print(f"  > ERROR: {err_dist:.4f} mm")
            print(f"  > PRECISIÓN: {precision:.2f}%")
            print("-" * 45)

    # Mosaico escalado
    ancho_etapa = 320 
    h, w = img.shape[:2]
    alto_etapa = int(h * (ancho_etapa / w))
    res1 = cv2.resize(img, (ancho_etapa, alto_etapa))
    res2 = cv2.resize(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), (ancho_etapa, alto_etapa))
    res3 = cv2.resize(color_img, (ancho_etapa, alto_etapa))
    mosaico = np.hstack((res1, res2, res3))

    # Configuración de ventana para que no se salga de pantalla
    nombre_ventana = f'Analisis: {nombre_imagen}'
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL) 
    cv2.imshow(nombre_ventana, mosaico)
    
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# ==========================================
# 3. EJECUCIÓN
procesar_figura(img09_entrada, img09_salida, 1.895, 0, 235, 360, 30, 25.0)
procesar_figura(img10a_entrada, img10a_salida, 1.25, 400, 312, 580, 0, 120.0)
procesar_figura(img10b_entrada, img10b_salida, 1.25, 400, 312, 580, 0, 90.0)
procesar_figura(img11_entrada, img11_salida, 0.5, 50, 50, 280, 0, -45.0, es_perno=True)
procesar_figura(img12_entrada, img12_salida, 1.0, -2252, -1728, -2000, -1800, 30.0)

print("Proceso finalizado.")