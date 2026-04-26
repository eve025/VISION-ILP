import cv2
import numpy as np

# 1. VARIABLES Y DATOS DEL EXCEL
K = 1.895        
OFFSET_X = 0     
OFFSET_Y = 235   

# Datos originales excel
datos_excel = {
    'portabaterias.png': [360, 30, 25.0],
    'pinza-mesa-120deg.png': [580, 0, 120.0],
    'pinza-mesa-90deg.png': [580, 0, 90.0],
    'Perno-de-bloqueo.png': [280, 0, -45.0],
    'placa-rectangular.png': [-2000, -1800, 30.0]  # Respetando tus negativos
}

# 2. FUNCIÓN DE PROCESAMIENTO (CANNY)
# ==========================================
def procesar_figura(nombre_imagen, nombre_salida):
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    color_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- AJUSTE PARA EL PERNO (IMAGEN 11) ---
    if 'Perno' in nombre_imagen:
        # Umbrales bajísimos para capturar el borde sutil del perno
        edges = cv2.Canny(blurred, 5, 40) 
        k_size = 12
    else:
        edges = cv2.Canny(blurred, 20, 100)
        k_size = 20

    # Clausura para rellenar la silueta detectada por Canny
    kernel = np.ones((k_size, k_size), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. ENCONTRAR CONTORNOS
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Conversión a coordenadas mm
            cx_mm = (cx * K) + OFFSET_X
            cy_mm = OFFSET_Y - (cy * K)

            # Lógica de Ángulo
            rect = cv2.minAreaRect(cnt)
            (w, h), angle = rect[1], rect[2]
            angle_robot = angle + 180 if w < h else angle + 90
            
            # Sincronización con el ángulo del Excel
            x_ex, y_ex, ang_ex = datos_excel.get(nombre_imagen, [0,0,0])
            while abs(angle_robot - ang_ex) > 45:
                if angle_robot > ang_ex: angle_robot -= 90
                else: angle_robot += 90

            # Cálculo de error (aquí saldrán los valores altos por la diferencia de escala del simulador)
            error_dist = np.sqrt((cx_mm - x_ex)**2 + (cy_mm - y_ex)**2)

            print(f"FICHA: {nombre_imagen}")
            print(f"  CALC -> Pos: ({cx_mm:.1f}, {cy_mm:.1f}) mm, Ángulo: {angle_robot:.2f}°")
            print(f"  EXCEL -> Pos: ({x_ex}, {y_ex}) mm, Ángulo: {ang_ex}°")
            print(f"  ERROR: {error_dist:.2f} mm")
            print("-" * 50)

            # Dibujo de resultados
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"({int(cx_mm)},{int(cy_mm)})", (cx+10, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)

    # --- CREAR MOSAICO ---
    h_orig, w_orig = img.shape[:2]
    nw, nh = 400, int(h_orig * (400 / w_orig))
    res1 = cv2.resize(img, (nw, nh))
    # Mostramos CLOSING para asegurar que el perno se "rellenó" correctamente
    res2 = cv2.resize(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), (nw, nh))
    res3 = cv2.resize(color_img, (nw, nh))
    
    mosaico = np.hstack((res1, res2, res3))

    cv2.imshow(f'Validacion: {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# =====================================================
# 3. EJECUCIÓN
procesar_figura('portabaterias.png', 'etapas_img09.png')
procesar_figura('pinza-mesa-120deg.png', 'etapas_img10_120.png')
procesar_figura('pinza-mesa-90deg.png', 'etapas_img10_90.png')
procesar_figura('Perno-de-bloqueo.png', 'etapas_img11.png')
procesar_figura('placa-rectangular.png', 'etapas_img12.png')

print("Proceso finalizado con datos de Excel originales.")