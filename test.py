import cv2
import face_recognition as fr

# Cargamos las imagenes
control_picture = fr.load_image_file("imgs/Keanu_Reeves_3.jpg")
test_picture = fr.load_image_file("imgs/Keanu_Reeves_1.webp")

SCALE = 0.5

# Cambiamos el espacio de color
control_picture = cv2.cvtColor(control_picture, cv2.COLOR_BGR2RGB)
test_picture = cv2.cvtColor(test_picture, cv2.COLOR_BGR2RGB)

# Escalamos las fotos
control_picture = cv2.resize(control_picture, None, fx=SCALE, fy=SCALE)
test_picture = cv2.resize(test_picture, None, fx=SCALE, fy=SCALE)

# Localizamos la cara en la foto A
face_A_location =  fr.face_locations(control_picture)[0]
coded_face_A = fr.face_encodings(control_picture)[0]

# Localizamos la cara en la foto B
face_B_location =  fr.face_locations(test_picture)[0]
coded_face_B = fr.face_encodings(test_picture)[0]

# Dibujamos el rectangulo en la foto A
cv2.rectangle(control_picture, 
              (face_A_location[3], face_A_location[0]), 
              (face_A_location[1], face_A_location[2]),
              (0,255,0),
              2)

# Dibujamos el rectangulo en la foto A
cv2.rectangle(test_picture, 
              (face_B_location[3], face_B_location[0]), 
              (face_B_location[1], face_B_location[2]),
              (0,255,0),
              2)

# print(face_A_location)

# Comparamos las caras
# (function) def compare_faces(
#     known_face_encodings: Any,
#     face_encoding_to_check: Any,
#     tolerance: float = 0.6
# ) -> list[Any]
result = fr.compare_faces([coded_face_A], coded_face_B, 0.5)
print(result)

distance = fr.face_distance([coded_face_A], coded_face_B)
print(distance)

# Mostrar resultados
cv2.putText(test_picture, 
            f"Result:{result}, distance:{distance.round(2)}",
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255,255,255),
            2)

# Mostrar en pantalla las imagenes
cv2.imshow("control_picture", control_picture)
cv2.imshow("test_picture", test_picture)
cv2.waitKey(0)