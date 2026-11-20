print("Alan Romero NC 0124")
import cv2
import os

# 1. Definir la ruta de la imagen usando la estructura actual de tu carpeta
ruta_imagen = 'images.jpg'

# Verificar que la imagen exista antes de procesarla
if not os.path.exists(ruta_imagen):
    print(f"Error: No se encontró el archivo '{ruta_imagen}' en el directorio.")
    exit()

# Datos para el encabezado de las ventanas
datos_alumno = "Alan Romero NC 0124"

# 2. Leer y mostrar la imagen original
img = cv2.imread(ruta_imagen)
cv2.imshow(f'{datos_alumno} - Imagen Original', img)

# 3. Obtener dimensiones (Alto, Ancho, Canales)
alto, ancho, canales = img.shape
print(f"Dimensiones originales: {ancho}x{alto} píxeles | Canales: {canales}")

# 4. Convertir a Escala de Grises
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow(f'{datos_alumno} - Escala de Grises', gris)

# 5. Redimensionar la imagen (50% de su tamaño original)
nuevo_ancho = int(ancho * 0.5)
nuevo_alto = int(alto * 0.5)
redimensionada = cv2.resize(img, (nuevo_ancho, nuevo_alto))
cv2.imshow(f'{datos_alumno} - Redimensionada (50%)', redimensionada)

# 6. Rotar la imagen (90 grados en sentido horario)
rotada = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow(f'{datos_alumno} - Rotada 90 grados', rotada)

# 7. Aplicar desenfoque (Gaussian Blur)
desenfoque = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow(f'{datos_alumno} - Desenfoque Gaussiano', desenfoque)

# 8. Detección de bordes (Canny)
bordes = cv2.Canny(gris, 100, 200)
cv2.imshow(f'{datos_alumno} - Deteccion de Bordes (Canny)', bordes)

# 9. Umbralización (Thresholding - Blanco y negro binario)
_, umbralizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
cv2.imshow(f'{datos_alumno} - Imagen Umbralizada', umbralizada)

# 10. Dibujar un cuadrado / rectángulo sobre una copia de la imagen
img_cuadrado = img.copy()
inicio_x, inicio_y = int(ancho * 0.25), int(alto * 0.25)
fin_x, fin_y = int(ancho * 0.75), int(alto * 0.75)
cv2.rectangle(img_cuadrado, (inicio_x, inicio_y), (fin_x, fin_y), (0, 255, 0), 3)
cv2.imshow(f'{datos_alumno} - Imagen con Cuadrado', img_cuadrado)

# 11. Detección y dibujo de contornos
contornos, _ = cv2.findContours(umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contornos = img.copy()
# Dibuja todos los contornos encontrados (-1) en color azul/rojo o verde (BGR: 255, 0, 0) con grosor 2
cv2.drawContours(img_contornos, contornos, -1, (255, 0, 0), 2)
print(f"Total de contornos detectados: {len(contornos)}")
cv2.imshow(f'{datos_alumno} - Deteccion de Contornos', img_contornos)

# 12. Guardar una de las imágenes procesadas en la misma carpeta
cv2.imwrite('resultado_bordes.jpg', bordes)
print("Se ha guardado 'resultado_bordes.jpg' en tu carpeta de proyecto.")

# Esperar a que presiones cualquier tecla para cerrar todas las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
