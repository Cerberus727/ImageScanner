import cv2
import numpy as np

# Preprocess image (resize, grayscale, blur, edge detection)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 1280))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 50, 150)
    return img, gray, edges

# Get the largest 4-point contour (assumed to be the document/notebook)
def get_document_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                best_contour = approx
                max_area = area
    return best_contour

# Warp the perspective to crop the notebook
def warp_perspective(img, contour):
    if contour is None:
        print("No document found!")
        return img
    contour = contour.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    s = contour.sum(axis=1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (int(width), int(height)))
    return warped

# Main pipeline

img_path = input("Enter path ")
img, gray, edges = preprocess_image(img_path)
document_contour = get_document_contour(edges)
cropped_notebook = warp_perspective(img, document_contour)

# Display only the cropped notebook without bounding boxes
cv2.imshow("Cropped Notebook", cropped_notebook)
cv2.waitKey(0)
cv2.destroyAllWindows()
