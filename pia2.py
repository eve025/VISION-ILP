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
def procesar_figura(nombre_imagen, nombre_salida):
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    color_img = img.copy()

    # 2. Detección de bordes (Canny)
    # edges = cv2.Canny(blurred, 50, 150) # Umbrales un poco más altos para limpiar ruido
    # edges = cv2.Canny(blurred, 8, 50) # Umbrales  muy bajos
    edges = cv2.Canny(blurred, 20, 100) # Umbrales  ajustados 
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Clausura para unir los bordes de la pieza en una sola masa
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. Encontrar contornos
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. FILTRO: Solo procesar el contorno más grande (la pieza principal)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        
        # CÁLCULO DE MOMENTOS
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # Centroide (Centro de Gravedad)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Dibujar Centroide
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"CG:({cx},{cy})", (cx+10, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Cálculo de Orientación
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            
            # Ajuste de ángulo , el eje para alinear el sistema de la imagen con el simulador
            if rect[1][0] < rect[1][1]:
                angle_robot = angle + 180
            else:
                angle_robot = angle + 90

            # Dibujar Caja y Vector de inclinación
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)
            
            length = 50
            rad = np.deg2rad(angle_robot)
            cv2.line(color_img, (cx, cy), 
                     (int(cx + length * np.cos(rad)), int(cy + length * np.sin(rad))), 
                     (0, 255, 0), 2)

            print(f"Imagen: {nombre_imagen:25} -> Centro: ({cx:3}, {cy:3}), Angulo: {angle_robot:6.2f} deg")

    # --- CREAR MOSAICO ---
    h, w = img.shape[:2]
    nueva_w = 400
    nueva_h = int(h * (nueva_w / w))
    
    res1 = cv2.resize(img, (nueva_w, nueva_h))
    res2 = cv2.resize(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), (nueva_w, nueva_h))
    res3 = cv2.resize(color_img, (nueva_w, nueva_h))
    mosaico = np.hstack((res1, res2, res3))

    cv2.imshow(f'Procesando: {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# 3. EJECUCIÓN
procesar_figura(img09_entrada, img09_salida)
procesar_figura(img10a_entrada, img10a_salida)
procesar_figura(img10b_entrada, img10b_salida)
procesar_figura(img11_entrada, img11_salida)
procesar_figura(img12_entrada, img12_salida)

print("Proceso finalizado.")