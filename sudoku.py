import cv2
import numpy
from collections import Counter

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

DATA_DIRECTORY = 'training_data/'
TEST_DATA_FILENAME = DATA_DIRECTORY + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIRECTORY + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIRECTORY + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIRECTORY + 'train-labels.idx1-ubyte'

def read_images(filename):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_columns = int.from_bytes(f.read(4), 'big')
        for image_index in range(10000):

            image = []
            for row_index in range(n_rows*n_columns):
                pixel = int.from_bytes(f.read(1), "big")
                #print(pixel)
                # if pixel != 0:
                #     pixel = 255
                image.append(pixel)
            images.append(image)
        return images

def read_labels(filename):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_labels = int.from_bytes(f.read(4), 'big')
        for label_index in range(10000):
            #print("a")
            label = int.from_bytes(f.read(1), "big")
            labels.append(label)
        return labels

def distance(x, y):
    #print(sum([(int.from_bytes(x_i, 'big') - int.from_bytes(y_i, 'big')) ** 2 for x_i, y_i in zip(x, y)]))
    return sum([(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]) ** 0.5

def get_training_distances_for_test_sample(X_train, test_sample):
    return[distance(train_sample, test_sample) for train_sample in X_train]


from collections import Counter


def knn(X_train, Y_train, X_test, k=9):
    Y_pred = []
    for test_sample_index, test_sample in enumerate(X_test):
        if numpy.count_nonzero(test_sample) < 50:  # skip mostly empty cells
            Y_pred.append(0)
            continue
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = sorted(range(len(training_distances)), key=lambda i: training_distances[i])
        candidates = [Y_train[i] for i in sorted_distance_indices[:k]]

        most_common = Counter(candidates).most_common(1)[0][0]
        Y_pred.append(most_common)

        print(f'Cell {test_sample_index}: Predicted = {most_common}, Nearest = {candidates}')
    return Y_pred


def main():
    image = cv2.imread('sudoku.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = numpy.zeros((gray.shape), numpy.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    div = numpy.float32(gray) / (close)
    res = numpy.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    # Finding the sudoku square contour
    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # Approximate the contour to a polygon with fewer vertices
    peri = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.02 * peri, True)  # 2% of perimeter tolerance

    if len(approx) == 4:
        # Found 4 corners, do perspective correction
        pts = approx.reshape(4, 2)

        def order_points(pts):
            rect = numpy.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[numpy.argmin(s)]  # top-left
            rect[2] = pts[numpy.argmax(s)]  # bottom-right

            diff = numpy.diff(pts, axis=1)
            rect[1] = pts[numpy.argmin(diff)]  # top-right
            rect[3] = pts[numpy.argmax(diff)]  # bottom-left
            return rect

        rect = order_points(pts)

        widthA = numpy.linalg.norm(rect[2] - rect[3])
        widthB = numpy.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))

        heightA = numpy.linalg.norm(rect[1] - rect[2])
        heightB = numpy.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = numpy.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(res, M, (maxWidth, maxHeight))
    else:
        # fallback: bounding box crop + resize
        x, y, w, h = cv2.boundingRect(best_cnt)
        warped = cv2.resize(res[y:y + h, x:x + w], (800, 800))

    # Use warped image for further processing
    # Finding horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dx = cv2.Sobel(warped, cv2.CV_16S, 0, 1)
    dx = cv2.convertScaleAbs(dx)
    ret, dx = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dx = cv2.morphologyEx(dx, cv2.MORPH_DILATE, horizontal_kernel)
    contours, hierarchy = cv2.findContours(dx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w / h > 6:
            cv2.drawContours(dx, [contour], 0, 255, -1)
        else:
            cv2.drawContours(dx, [contour], 0, 0, -1)
    dx = cv2.morphologyEx(dx, cv2.MORPH_CLOSE, None, iterations=2)

    # Finding vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    dy = cv2.Sobel(warped, cv2.CV_16S, 1, 0)
    dy = cv2.convertScaleAbs(dy)
    ret, dy = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dy = cv2.morphologyEx(dy, cv2.MORPH_DILATE, vertical_kernel)
    contours, hierarchy = cv2.findContours(dy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 7:
            cv2.drawContours(dy, [contour], 0, 255, -1)
        else:
            cv2.drawContours(dy, [contour], 0, 0, -1)
    dy = cv2.morphologyEx(dy, cv2.MORPH_CLOSE, None, iterations=2)

    # Combining the two images to get the grid mask
    final_image = cv2.bitwise_and(dx, dy)
    contours, hierarchy = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 3 or w / h > 3:
            cv2.drawContours(final_image, [contour], 0, 0, -1)
        else:
            cv2.drawContours(final_image, [contour], 0, 255, -1)

    # Finding centroids of grid intersections
    contours, hierarchy = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gridPoints = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        else:
            point = [0, 0]
        gridPoints.append(point)

    # Sort the grid points vertically, then horizontally
    gridPoints = sorted(gridPoints, key=lambda x: x[1])
    print(f"Detected grid points: {len(gridPoints)}")
    gridPoints2D = numpy.zeros((10, 10, 2))
    for i in range(10):
        row_points = gridPoints[10 * i:10 * (i + 1)]
        row_points = sorted(row_points, key=lambda x: x[0])
        for j in range(10):
            gridPoints2D[i][j][0] = row_points[j][0]
            gridPoints2D[i][j][1] = row_points[j][1]

    # Extract individual cell images using perspective transform
    image_array = []
    for i in range(9):
        for j in range(9):
            pts1 = numpy.float32([
                [gridPoints2D[i][j][0], gridPoints2D[i][j][1]],
                [gridPoints2D[i][j + 1][0], gridPoints2D[i][j + 1][1]],
                [gridPoints2D[i + 1][j][0], gridPoints2D[i + 1][j][1]],
                [gridPoints2D[i + 1][j + 1][0], gridPoints2D[i + 1][j + 1][1]]
            ])
            pts2 = numpy.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M_cell = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(warped, M_cell, (300, 300))

            ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # dst = cv2.bitwise_not(dst)
            image_array.append(dst)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # smaller kernel
    for i in range(len(image_array)):
        image_array[i] = cv2.morphologyEx(image_array[i], cv2.MORPH_CLOSE, kernel, iterations=1)
        # Then crop margins a bit less aggressively
        image_array[i] = image_array[i][10:290, 10:290]
        image_array[i] = cv2.resize(image_array[i], (28, 28))

    # # Display each cell (optional)
    # for idx, img in enumerate(image_array):
    #     cv2.imshow(f'Cell {idx}', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Prepare test data
    X_test = [numpy.array(img).ravel() for img in image_array]

    # Invert pixel values for KNN if needed (black background, white digits)
    for i in range(len(X_test)):
        for j in range(len(X_test[i])):
            X_test[i][j] = 0 if X_test[i][j] <= 200 else 255

    # Load training data and labels
    X_train = read_images(TRAIN_DATA_FILENAME)
    Y_train = read_labels(TRAIN_LABELS_FILENAME)

    predictions = knn(X_train, Y_train, X_test, 9)

    # Format predictions as 9x9 grid
    print("\nPredicted Sudoku Grid:")
    for i in range(9):
        row = predictions[i * 9:(i + 1) * 9]
        print(" ".join(str(d) for d in row))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/