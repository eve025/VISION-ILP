import cv2
import numpy as np

# PARÁMETROS DE CALIBRACIÓN
K_PIXEL_MM = 0.25
OFF_X = -58.0
OFF_Y = 177.0
ANGULO_MESA_Y = -30.0 # --- Inclinación R(0, -30, 0) ---

# 1. DEFINICIÓN DE IMAGENES
img09_entrada, img01_salida = '#1.png', 'etapas_img01.png'
img10_entrada, img02_salida = '#2.png', 'etapas_img02.png'
img11_entrada, img03_salida = '#3.png', 'etapas_img03.png'
img12_entrada, img04_salida = '#4.png', 'etapas_img04.png'

num_foto = 1

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO 
def procesar_figura(nombre_imagen, nombre_salida, x_real, y_real, ang_real):
    global num_foto
    img_raw = cv2.imread(nombre_imagen)
    if img_raw is None:
        print(f"No se encontró: {nombre_imagen}")
        return
    
    img = cv2.resize(img_raw, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_img = img.copy()

    low, high = (7, 7)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((5, 5), np.uint8) 
    dilated = cv2.dilate(edges, kernel, iterations=1) 
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt) 
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            cx_robot = (cx * K_PIXEL_MM) + OFF_X
            cy_robot = OFF_Y - (cy * K_PIXEL_MM)

            # --- ORIENTACIÓN ---
            rect = cv2.minAreaRect(cnt)
            (x_centro, y_centro), (ancho, alto), angle = rect

            if ancho < alto:
                angle_robot = angle + 180
            else:
                angle_robot = angle + 90
            
            # --- CÁLCULO DE ORIENTACIÓN DE TRABAJO (TCP) ---
            # Se combina la orientación del objeto con la inclinación de la mesa
            orientacion_trabajo = angle_robot + ANGULO_MESA_Y

            err_dist = np.sqrt((cx_robot - x_real)**2 + (cy_robot - y_real)**2)
            precision = (1 - (err_dist / 400)) * 100 

            # --- DIBUJAR ---
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos_coord = (cx + 15, cy - 15)
            pos_ang = (cx + 15, cy + 25)

            cv2.putText(color_img, f"({int(cx_robot)},{int(cy_robot)})", pos_coord, font, 1.1, (255, 255, 255), 2)
            cv2.putText(color_img, f"{angle_robot:.1f}deg", pos_ang, font, 1.1, (0, 255, 255), 2)

            # --- FORMATO TXT EN CONSOLA ---
            print(f"Imagen {num_foto}: {nombre_imagen}")
            print(f"  > Píxeles: ({cx}, {cy})")
            print(f"  > Centroide Calc: X={cx_robot:.2f}, Y={cy_robot:.2f}, Ang={angle_robot:.1f}°")
            print(f"  > ORIENTACIÓN DE TRABAJO (TCP): {orientacion_trabajo:.1f}°") # --- NUEVO ---
            print(f"  > Centroide Real: X={x_real}, Y={y_real}, Ang={ang_real:.1f}°")
            print(f"  > ERROR DISTANCIA: {err_dist:.4f} mm")
            print(f"  > PRECISIÓN: {precision:.2f}%")
            print("-" * 45)
            
            num_foto += 1

    ancho_etapa = 640
    h, w = img.shape[:2]
    alto_etapa = int(h * (ancho_etapa / w))
    res1 = cv2.resize(img, (ancho_etapa, alto_etapa))
    res2 = cv2.resize(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), (ancho_etapa, alto_etapa))
    res3 = cv2.resize(color_img, (ancho_etapa, alto_etapa))
    mosaico = np.hstack((res1, res2, res3))
    
    cv2.namedWindow(f'Analisis: {nombre_imagen}', cv2.WINDOW_NORMAL) 
    cv2.imshow(f'Analisis: {nombre_imagen}', mosaico)
    cv2.imwrite(nombre_salida, mosaico)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# ==========================================
# 3. EJECUCIÓN (VALIDACIÓN CON CARTESIAN JOG)

# IMAGEN 01: Engranaje
# Robot marcó: X=581.699, Y=, 98.84
# Real_X = 581.7 - 500 = 112.7
# Real_Y = 98.84 - 0 = 98.84
#(581.7 Robot_X - 500 Offset) y (0 Offset + 98.84 Robot_Y) 
procesar_figura(img09_entrada, img01_salida, 81.7, 98.84, 0.0) #real  


# IMAGEN 2: Bracket
# Robot marcó: X=582.281, Y=98.84
# Real_X = 582.3 - 500 = 250.3
# Real_Y = 98.84 - 0 = 149.3
#(582.3 Robot_X - 500 Offset) y (100 Offset + 98.84 Robot_Y) 
procesar_figura(img10_entrada, img02_salida, 82.28, 98.84, 45.0)  #real  


# IMAGEN 11: Eslabón 
# Robot marcó: X=484.548, Y=98.84
# Real_X = 484.54 - 500 = -15.45
# Real_Y = 98.84 - 0 = 98.84
#(484.6 Robot_X - 500 Offset) y (0 Offset + 98.84 Robot_Y)
procesar_figura(img11_entrada, img03_salida, -15.45, 98.84, 0.0)  #real  


# IMAGEN 12: Flange
# Robot marcó: X=639.099, Y=98.84
# Real_X = 639.099 - 500 = 139.099
# Real_Y = 98.84 - 0 = 98.84
# (639.1 Robot_X - 200 Offset) y (0 Offset + 98.84 Robot_Y) 
procesar_figura(img12_entrada, img04_salida, 139.10, 98.84, 0.0)  #real  


print("Proceso finalizado.")