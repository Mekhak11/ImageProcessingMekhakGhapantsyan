import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def morphological_filtering(edges):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def fit_ellipses(contours):
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    return ellipses

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def find_oval(image_path, filename):
    image = cv2.imread(image_path)

    preprocessed_image = preprocess_image(image)

    edges = edge_detection(preprocessed_image)

    filtered_edges = morphological_filtering(edges)

    contours = find_contours(filtered_edges)

    ellipses = fit_ellipses(contours)

    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        largest_ellipse = None
        max_area = 0

        for ellipse in ellipses:
            center, axes, angle = ellipse
            if (x <= center[0] <= x + w) and (y <= center[1] <= y + h):
                area = np.pi * axes[0] * axes[1] / 4
                if area > max_area:
                    max_area = area
                    largest_ellipse = ellipse

        if largest_ellipse:
            center, axes, angle = largest_ellipse

            top_left_x = int(center[0] - axes[0] / 2)
            top_left_y = int(center[1] - axes[1] / 2)
            width = int(axes[0])
            height = int(axes[1])

            print(
                f'image: {filename} TopLeftCoordinate X: {top_left_x}, TopLeftCoordinate Y: {top_left_y}, Width: {width}, Height: {height}')
            cv2.ellipse(image, largest_ellipse, (0, 255, 0), 2)

    cv2.imshow('Detected Ovals', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    files = [
        '57-11',
        '58-11',
        '59-11',
        '60-11',
        '61-11',
        '62-11',
        '63-11',
        '64-11',
    ]
    for file_name in files:
        find_oval(f"../Homework2/mask/{file_name}.png",file_name)

# Example usage
main()