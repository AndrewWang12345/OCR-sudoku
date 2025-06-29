import cv2
import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from collections import Counter
import time
# Read CUDA kernel source
with open("kernels.cu", "r") as f:
    kernel_code = f.read()

# Compile with PyCUDA
mod = SourceModule(kernel_code)

# Get function references
morphology_dilation = mod.get_function("morphology_dilation")
morphology_erosion = mod.get_function("morphology_erosion")
knn_distance = mod.get_function("knn_distance")
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

DATA_DIRECTORY = 'training_data/'
TEST_DATA_FILENAME = DATA_DIRECTORY + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIRECTORY + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIRECTORY + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIRECTORY + 'train-labels.idx1-ubyte'
def is_valid(board, row, col, num):
    # Check row
    if any(board[row][c] == num for c in range(9)):
        return False
    # Check column
    if any(board[r][col] == num for r in range(9)):
        return False
    # Check 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if board[r][c] == num:
                return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False  # No valid number found, backtrack
    return True  # Solved

def print_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

def gpu_morphology_close(gray, kernel):
    height, width = gray.shape

    # Flatten and copy kernel mask to constant memory
    mask_flat = kernel.flatten().astype(np.uint8)
    d_mask, _ = mod.get_global('d_mask')
    cuda.memcpy_htod(d_mask, mask_flat)

    # Allocate GPU buffers
    input_gpu = cuda.mem_alloc(gray.nbytes)
    dilated_gpu = cuda.mem_alloc(gray.nbytes)
    eroded_gpu = cuda.mem_alloc(gray.nbytes)

    cuda.memcpy_htod(input_gpu, gray)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    # Run dilation
    morphology_dilation(input_gpu, dilated_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    # Run erosion on dilated image (closing)
    morphology_erosion(dilated_gpu, eroded_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    # Copy result back to CPU
    gpu_closed = np.empty_like(gray)
    cuda.memcpy_dtoh(gpu_closed, eroded_gpu)

    return gpu_closed
def read_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # skip the header
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(-1, 28 * 28)
    return images

def read_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # skip the header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def distance(x, y):
    #print(sum([(int.from_bytes(x_i, 'big') - int.from_bytes(y_i, 'big')) ** 2 for x_i, y_i in zip(x, y)]))
    return sum([(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)])

def get_training_distances_for_test_sample(X_train, test_sample):
    return[distance(train_sample, test_sample) for train_sample in X_train]

# [6005030.0, 6815366.0, 3918958.0, 3862798.0, 5088761.0, 6339780.0, 4070545.0, 8058638.0, 2386239.0, 4691529.0]
#  [5585555.0, 7660181.0, 7166893.0, 6652243.0, 6138596.0, 6726105.0, 6611110.0, 6609983.0, 6030444.0, 6747594.0]

def knn(X_train, Y_train, X_test, k=9):
    Y_pred = []

    # Convert data to float32 numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    distances = np.zeros((len(X_test), len(X_train)), dtype=np.float32)

    # Allocate GPU memory
    d_X_train = cuda.mem_alloc(X_train.nbytes)
    d_X_test = cuda.mem_alloc(X_test.nbytes)
    d_distances = cuda.mem_alloc(distances.nbytes)

    # Copy to GPU
    cuda.memcpy_htod(d_X_train, X_train)
    cuda.memcpy_htod(d_X_test, X_test)

    # Set launch config
    block_size = (16, 16, 1)
    grid_y = (len(X_test) + 15) // 16
    grid_x = (len(X_train) + 15) // 16
    # Call CUDA kernel
    knn_distance(d_X_train, d_X_test, d_distances,
                 np.int32(len(X_train)), np.int32(len(X_test)), np.int32(X_train.shape[1]),
                 block=block_size, grid=(grid_x, grid_y))

    # Copy distances back
    cuda.memcpy_dtoh(distances, d_distances)
    # Do the top-k voting on CPU
    for test_sample_index in range(len(X_test)):
        if np.count_nonzero(X_test[test_sample_index]) < 70:
            Y_pred.append(0)
            continue

        dist_row = distances[test_sample_index]
        sorted_indices = np.argsort(dist_row)[:k]
        candidates = [Y_train[i] for i in sorted_indices[:k]]
        most_common = Counter(candidates).most_common(1)[0][0]
        Y_pred.append(most_common)
    return Y_pred


def main():
    image = cv2.imread('sudoku.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create elliptical kernel same as OpenCV uses for comparison
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)).astype(np.uint8)
    close = gpu_morphology_close(gray, kernel)

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
        image_array[i] = image_array[i][50:250, 50:250]
        image_array[i] = cv2.resize(image_array[i], (28, 28))

    # # Display each cell (optional)
    # for idx, img in enumerate(image_array):
    #     cv2.imshow(f'Cell {idx}', cv2.resize(img, (280, 280)))
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

    # Convert predictions (list) to 9x9 grid (list of lists of ints)
    sudoku_grid = [predictions[i * 9:(i + 1) * 9] for i in range(9)]

    if solve_sudoku(sudoku_grid):
        print("\nSolved Sudoku Grid:")
        print_board(sudoku_grid)
    else:
        print("Unsolvable")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/