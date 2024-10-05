import cv2
import os
import numpy as np

image_path = "./rawdata/Compressed Nepali Handwritten Data_48 (1).jpg"
outputdir = "annotateddata"

def show(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generateContours(image : cv2.typing.MatLike):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    inverted_thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ncounturs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 20<w<100 and 20<h<100: ncounturs.append(c)
    return ncounturs

def highlightContours(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show(image, "contours")


def saveContours(image, contours, dirname):
    os.makedirs(dirname, exist_ok=True)
    cell_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell = image[y:y+h, x:x+w]
        cell_filename = f"{dirname}/cell_{cell_count}.png"
        cv2.imwrite(cell_filename, cell)
        cell_count += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Total cells extracted: {cell_count}")

image = cv2.imread(image_path)
nheight = 800
nwidth = image.shape[1] * nheight // image.shape[0]
resized = cv2.resize(image, (nwidth, nheight))

contours = generateContours(resized)
highlightContours(resized, contours)
#saveContours(image, contours, outputdir)

