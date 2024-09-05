import cv2
import face_recognition as fr
import os
import numpy as np

path = "persons"
persons_images = []
persons_names = []
persons_list = os.listdir(path)
encoded_person_list = []

for name in persons_list:
    # Leemos imagen y lo agregamos en la lista de imagenes
    image = cv2.imread(f"{path}\\{name}")
    persons_images.append(image)

    # Agregamos el nombre de la persona a una lista
    # splitext(name) | ejem. "karl-mark.jpg" => ('karl-marx', '.jpg')
    persons_names.append(os.path.splitext(name)[0])

    # Codificamos imagen y lo agregamos a una lista
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoded_image = fr.face_encodings(image_RGB)[0]
    encoded_person_list.append(encoded_image)


print(persons_names)

# encoded_person_list = encode(persons_images)

# Capturamos una imagen con la webcam
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
success, image_captured = capture.read()

if not success:
    print("No se pudo hacer la captura")
else:
    # cv2.imshow("test", image_captured)
    capture_face = fr.face_locations(image_captured)
    encoded_capture_face = fr.face_encodings(image_captured, capture_face)
    
    for face, location_face in zip(encoded_capture_face, capture_face):
        # Comparamos medidas con las imagenes
        matches = fr.compare_faces(encoded_person_list, face)
        distances = fr.face_distance(encoded_person_list, face)
        print(distances)

        # Buscamos con quien de las imagenes guardadas coincide mas
        match_index = np.argmin(distances)  
        if distances[match_index] > 0.6:
            print("No existen coincidencias")
        else:
            person_name = persons_names[match_index]

            y1, x2, y2, x1 = location_face
            cv2.rectangle(image_captured, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(image_captured, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(image_captured, person_name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            cv2.imshow("Image", image_captured)
    
        cv2.waitKey(0)