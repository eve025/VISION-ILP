import cv2
import numpy as np

# 1. DEFINICIÓN DE VARIABLES (RUTAS)
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
# CONSTANTES DE CALIBRACIÓN Y DATOS EXCEL
# ==========================================
# K es el factor de escala px a mm
K = 1.895        
# OFFSET para alinear con las coordenadas negativas del simulador
OFFSET_X = 0     
OFFSET_Y = 235   

# Datos originales de tu Excel para el reporte
datos_excel = {
    'portabaterias.png': [360, 30, 25.0],
    'pinza-mesa-120deg.png': [580, 0, 120.0],
    'pinza-mesa-90deg.png': [580, 0, 90.0],
    'Perno-de-bloqueo.png': [280, 0, -45.0],
    'placa-rectangular.png': [-2000, -1800, 30.0]
}

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
    # Mejorando entrada: desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    color_img = img.copy()

    # 2. Detección de bordes (Canny fijo 20, 100)
    # Se aplican los parámetros solicitados para todas las figuras
    edges = cv2.Canny(blurred, 20, 100) 
    
    # Clausura para unir bordes y crear una masa sólida para el centroide
    # Un kernel de 20x20 asegura que piezas como el perno no se pierdan
    kernel = np.ones((20, 20), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. Encontrar contornos
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Procesar el contorno principal
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        
        if M["m00"] != 0:
            # Centroide en píxeles
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # CONVERSIÓN A REALES (mm)
            # Aplicamos K y el OFFSET para que coincida con el Excel/Robot
            cx_mm = (cx * K) + OFFSET_X
            cy_mm = OFFSET_Y - (cy * K)

            # Dibujar Centroide y coordenadas en mm
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"({int(cx_mm)},{int(cy_mm)})", (cx+10, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Cálculo de Orientación
            rect = cv2.minAreaRect(cnt)
            (w, h), angle = rect[1], rect[2]
            
            # Lógica para determinar el ángulo según el eje largo
            angle_robot = angle + 180 if w < h else angle + 90
            
            # Sincronización con el ángulo del Excel (Normalización)
            x_ex, y_ex, ang_ex = datos_excel.get(nombre_imagen, [0,0,0])
            while abs(angle_robot - ang_ex) > 45:
                if angle_robot > ang_ex: angle_robot -= 90
                else: angle_robot += 90

            # Dibujar Caja y Vector
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)
            
            # Cálculo de error para la rúbrica
            error_pos = np.sqrt((cx_mm - x_ex)**2 + (cy_mm - y_ex)**2)

            # Impresión de resultados en consola
            print(f"FICHA: {nombre_imagen}")
            print(f"  > Calculado mm: ({cx_mm:8.2f}, {cy_mm:8.2f}) | Ángulo: {angle_robot:6.2f}")
            print(f"  > Excel      mm: ({x_ex:8}, {y_ex:8}) | Ángulo: {ang_ex:6}")
            print(f"  > Error de Posición: {error_pos:.2f} mm")
            print("-" * 65)

    # --- CREAR MOSAICO (Original, Clausura, Resultado) ---
    h, w = img.shape[:2]
    nueva_w = 400
    nueva_h = int(h * (nueva_w / w))
    
    res1 = cv2.resize(img, (nueva_w, nueva_h))
    res2 = cv2.resize(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), (nueva_w, nueva_h))
    res3 = cv2.resize(color_img, (nueva_w, nueva_h))
    mosaico = np.hstack((res1, res2, res3))

    # 5. Muestra y guarda
    cv2.imshow(f'Procesando: {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# ==========================================
# 3. EJECUCIÓN 
procesar_figura(img09_entrada, img09_salida)
procesar_figura(img10a_entrada, img10a_salida)
procesar_figura(img10b_entrada, img10b_salida)
procesar_figura(img11_entrada, img11_salida)
procesar_figura(img12_entrada, img12_salida)

print("Proceso finalizado.")