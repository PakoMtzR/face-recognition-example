import cv2
import face_recognition as fr
import os
import numpy as np
from pathlib import Path

path = Path(os.getcwd(), "persons")  # "persons"
persons_images = []
persons_names = []
persons_list = os.listdir(path)
encoded_person_list = []

# Obtenemos la lista de personas y sus imagenes
for name in persons_list:
    # Leemos imagen y lo agregamos en la lista de imagenes
    image = cv2.imread(f"{path}\\{name}")
    persons_images.append(image)

    # Agregamos el nombre de la persona a una lista
    # splitext(name) | ejem. "karl-mark.jpg" => ('karl-marx', '.jpg')
    persons_names.append(os.path.splitext(name)[0])

    # Codificamos imagen y lo agregamos a una lista
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Cambiamos espacio de color a RGB
    encoded_image = fr.face_encodings(image_RGB)[0]     # Codificamos la cara
    encoded_person_list.append(encoded_image)           # Lo guardamos en una lista

# Imprimimos por consola los nombres de las personas en el banco de imagenes
print("Registered persons:")
print("-"*30)
for i,person in enumerate(persons_names):
    print(f"{i+1}\t{person}")
print("-"*30)

# Webcam
camara = cv2.VideoCapture(0)

while True:
    success, frame = camara.read()
    
    capture_face = fr.face_locations(frame)
    encoded_capture_face = fr.face_encodings(frame, capture_face)

    for face, location_face in zip(encoded_capture_face, capture_face):
        # Comparamos medidas con las imagenes
        matches = fr.compare_faces(encoded_person_list, face)
        distances = fr.face_distance(encoded_person_list, face)
        # print(distances)

        # Buscamos con quien de las imagenes guardadas coincide mas
        match_index = np.argmin(distances)  
        if distances[match_index] <= 0.6:
            person_name = persons_names[match_index]

            y1, x2, y2, x1 = location_face
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, person_name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Camara (press 'q' to close)", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
camara.release()
cv2.destroyAllWindows()