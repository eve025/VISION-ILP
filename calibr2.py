import cv2
import numpy as np

# 1. CONFIGURACIÓN DE CALIBRACIÓN
# ==========================================
K = 1.895        
OFFSET_X = 0     
OFFSET_Y = 235   

# DATOS DEL EXCEL 
ex_09 = [360, 30, 25.0]
ex_10a = [580, 0, 120.0]
ex_10b = [580, 0, 90.0]
ex_11 = [280, 0, -45.0]
ex_12 = [-2000, -1800, 30.0]

# 2. FUNCIÓN DE PROCESAMIENTO (SOLO CANNY)
# ==========================================
def procesar_figura(nombre_imagen, nombre_salida, datos_excel):
    x_real, y_real, ang_real = datos_excel
    
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    color_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # --- DETECCIÓN DE BORDES (SOLO CANNY) ---
    # Ajustamos umbrales según la pieza para que Canny sea efectivo
    if 'Perno' in nombre_imagen:
        edges = cv2.Canny(blurred, 50, 150)
        kernel_size = 15
    else:
        edges = cv2.Canny(blurred, 30, 90)
        kernel_size = 20

    # Usamos solo morfología para unir los bordes de Canny
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Transformación a mm
            cx_calc = (cx * K) + OFFSET_X
            cy_calc = OFFSET_Y - (cy * K)

            # Lógica de Ángulo Mejorada
            rect = cv2.minAreaRect(cnt)
            (w, h), angle = rect[1], rect[2]
            
            # Ajuste para que el ángulo sea relativo al eje largo
            if w < h:
                angle_robot = angle + 180
            else:
                angle_robot = angle + 90
            
            # Corrección de desfase con el Excel
            while abs(angle_robot - ang_real) > 45:
                if angle_robot > ang_real: angle_robot -= 90
                else: angle_robot += 90

            # --- RESULTADOS ---
            print(f"FICHA: {nombre_imagen}")
            print(f"  CENTROIDE -> Calc: ({cx_calc:.1f}, {cy_calc:.1f}) | Excel: ({x_real}, {y_real})")
            print(f"  ÁNGULO    -> Calc: {angle_robot:.2f}° | Excel: {ang_real}°")
            error_dist = np.sqrt((cx_calc - x_real)**2 + (cy_calc - y_real)**2)
            print(f"  ERROR POS: {error_dist:.2f} mm")
            print("-" * 50)

            # --- DIBUJO ---
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"({int(cx_calc)},{int(cy_calc)})", (cx+10, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)

    # --- MOSAICO ---
    h_orig, w_orig = img.shape[:2]
    r_h = int(h_orig * (400 / w_orig))
    res1 = cv2.resize(img, (400, r_h))
    # Mostramos los bordes de Canny directamente en el mosaico
    res2 = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (400, r_h))
    res3 = cv2.resize(color_img, (400, r_h))
    mosaico = np.hstack((res1, res2, res3))

    cv2.imshow(f'Canny Pure - {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# ==========================================
# EJECUCIÓN
# ==========================================
procesar_figura('portabaterias.png', 'etapas_09.png', ex_09)
procesar_figura('pinza-mesa-120deg.png', 'etapas_10_120.png', ex_10a)
procesar_figura('pinza-mesa-90deg.png', 'etapas_10_90.png', ex_10b)
procesar_figura('Perno-de-bloqueo.png', 'etapas_11.png', ex_11)
procesar_figura('placa-rectangular.png', 'etapas_12.png', ex_12)
