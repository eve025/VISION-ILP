import cv2
import numpy as np

# --- nombre de la imagen a procesar ---
nombre_imagen = 'img01.png'           # nombre de la imagen a procesar
nombre_salida = 'salida_img01.png'    # Nombre de la imagen procesada a guardar

# 1. Cargar imagen en escala de grises
img = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # dibujar en color los resultados

# 2. Detección de bordes
edges = cv2.Canny(img, 50, 150)

# 3. Encontrar contornos en la imagen binaria
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Procesa cada contorno para calcular el centroide y la orientación
for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        # Centroide
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Dibujar centroide
        cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(color_img, f"({cx},{cy})", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Calcular orientación con rectángulo mínimo
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)  # Contorno en azul

        # Dibuja línea de orientación
        length = 40
        angle_rad = np.deg2rad(angle)
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        pt1 = (cx, cy)
        pt2 = (cx + dx, cy + dy)
        cv2.line(color_img, pt1, pt2, (0,255,0), 2)

        print(f"Centro: ({cx}, {cy}), Ángulo: {angle} grados")

# 5. Muestra y guarda la imagen procesada
cv2.imshow('Resultado', color_img)
cv2.imwrite(nombre_salida, color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()