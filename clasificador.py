import os
import cv2
import numpy as np

# Carpetas de origen y destino
source_folder = "fotos_vacas/fotos"
output_folders = {
    "vaca_de_pie": "fotos_vacas/vaca_de_pie",
    "vaca_acostada": "fotos_vacas/vaca_acostada",
    "cama_vacia": "fotos_vacas/cama_vacia",
}

# Crear carpetas de destino si no existen
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Obtener todas las imágenes de la carpeta de origen
image_files = [
    f for f in os.listdir(source_folder) if f.endswith((".png", ".jpg", ".jpeg"))
]

# Función para mostrar imágenes con instrucciones y moverlas a la carpeta correspondiente
def classify_images():
    history = []  # Para rastrear las imágenes movidas y sus ubicaciones anteriores
    current_image_index = 0  # Índice de la imagen actual

    while current_image_index < len(image_files):
        image_file = image_files[current_image_index]
        image_path = os.path.join(source_folder, image_file)
        img = cv2.imread(image_path)

        # Quitar la curvatura de la imagen
        pts1 = np.float32([[0, 0], [0, 1920], [1080, 0], [1920, 1080]])
        pts2 = np.float32([[0, 0], [0, 1850], [1080, 0], [1850, 1080]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (1680, 950))
        # Quita los primeros 225 pixeles de la imagen
        img = img[:, 250:]
        print(img.shape)
        # Divide la imagen en 3 partes
        imgs = [img[0:950, 0:450], img[0:950, 450:900], img[0:950, 900:1350]]
        c = 0
        for imagenes in imgs:
            c += 1
            while True:
                # Crear una copia de la imagen para mostrar las instrucciones
                img_with_instructions = imagenes.copy()
                # Contar las imágenes en cada carpeta
                count_vaca_de_pie = len(os.listdir(output_folders["vaca_de_pie"]))
                count_vaca_acostada = len(os.listdir(output_folders["vaca_acostada"]))
                count_cama_vacia = len(os.listdir(output_folders["cama_vacia"]))

                # Agregar el contador en la parte superior de la imagen
                counter_text = (
                    f"Imagen {current_image_index + 1} de {len(image_files)} | "
                    f"De pie: {count_vaca_de_pie} | "
                    f"Acostada: {count_vaca_acostada} | "
                    f"Cama vacia: {count_cama_vacia}"
                )
                y0, dy = 30, 30
                for i, line in enumerate(counter_text.split('|')):
                    y = y0 + i * dy
                    cv2.putText(
                        img_with_instructions,
                        line.strip(),
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Agregar la guía de teclas a la copia de la imagen con salto de línea y en la parte inferior
                instructions = (
                    "1: Vaca de pie\n2: Vaca acostada\n3: Cama vacia\nEsc: Salir\n4: Cancelar"
                )
                y0, dy = img_with_instructions.shape[0] - 100, 30
                for i, line in enumerate(instructions.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(
                        img_with_instructions,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Mostrar la copia de la imagen con las instrucciones
                cv2.imshow("Clasifica la imagen", img_with_instructions)
                key = cv2.waitKey(0)  # Esperar a que el usuario presione una tecla

                # Opciones de clasificación
                if key == ord("1"):
                    directory = f"fotos_vacas/vaca_de_pie/{c}_{image_file}"
                    # guardar la imagen recortada en la carpeta correspondiente
                    cv2.imwrite(directory, imagenes)
                    history.append((directory, current_image_index))
                    break
                elif key == ord("2"):
                    directory = f"fotos_vacas/vaca_acostada/{c}_{image_file}"
                    # guardar la imagen recortada en la carpeta correspondiente
                    cv2.imwrite(directory, imagenes)
                    history.append((directory, current_image_index))
                    break
                elif key == ord("3"):
                    directory = f"fotos_vacas/cama_vacia/{c}_{image_file}"
                    # guardar la imagen recortada en la carpeta correspondiente
                    cv2.imwrite(directory, imagenes)
                    history.append((directory, current_image_index))
                    break
                elif key == 27:  # Código ASCII para la tecla Esc
                    print("Saliendo del programa...")
                    return 0
                elif key == ord("4"):  # Código para cancelar la última clasificación
                    if history:
                        last_directory, last_index = history.pop()
                        print(f"Cancelando la clasificación de {last_directory}")
                        os.remove(last_directory)
                        current_image_index = last_index - 1  # Retroceder a la imagen anterior
                        break  # Salir del bucle interno para volver a mostrar la imagen anterior
                    else:
                        print("No hay imágenes para cancelar.")
                        cv2.destroyAllWindows()
                        continue
                # pressionar espacio para pasar omitir la imagen
                elif key == ord(" "):
                    print(f"Omitiendo {image_file}")
                    break
                else:
                    print(f"Clasificación inválida para {image_file}. Inténtalo de nuevo.")
                    cv2.destroyAllWindows()
                    continue
            # Cerrar la ventana de la imagen
            cv2.destroyAllWindows()

        current_image_index += 1  # Avanzar a la siguiente imagen

if __name__ == "__main__":
    classify_images()
