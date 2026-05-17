import cv2

def resize_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)

    image = cv2.resize(image, size)

    return image
  
