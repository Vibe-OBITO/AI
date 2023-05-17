import cv2
import numpy as np
import tensorflow as tf

# загружаем модель MobileNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# создаем функцию для классификации изображения
def classify_image(image):
    # изменяем размер изображения
    img = cv2.resize(image, (224, 224))
    # преобразуем изображение в формат, подходящий для модели
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    # делаем предсказание на модели
    predictions = model.predict(img)
    # получаем название предмета с наибольшей вероятностью
    results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return results[0][1]

# создаем объект VideoCapture для работы с камерой
cap = cv2.VideoCapture(0)

while True:
    # снимаем кадр с камеры
    ret, frame = cap.read()
    # классифицируем изображение
    object_name = classify_image(frame)
    # выводим название предмета на экран
    cv2.putText(frame, object_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # показываем кадр на экране
    cv2.imshow('Camera', frame)
    # выход из цикла при нажатии на клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
