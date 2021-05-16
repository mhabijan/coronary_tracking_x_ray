import numpy as np
from skimage.filters import meijering, threshold_otsu, frangi, gaussian, apply_hysteresis_threshold
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, canny, match_template
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.util import invert
from skimage.exposure import rescale_intensity
from PIL import Image
import copy


def _sortbyabs(array, axis=0):
    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(array).argsort(axis)

    # Return abs sorted array
    return array[tuple(index)]


def compute_hessian_eigenvalues(image, sigma):
    # Convert image to float
    image = img_as_float(image)

    # Make nD hessian
    hessian_elements = hessian_matrix(image, sigma=sigma, order='rc')

    # Correct for scale
    hessian_elements = [sigma * e for e in hessian_elements]

    # Compute Hessian eigenvalues
    hessian_eigenvalues = np.array(hessian_matrix_eigvals(hessian_elements))

    # Sort eigenvalues by absolute values in ascending order
    hessian_eigenvalues = _sortbyabs(hessian_eigenvalues, axis=0)
    eigen_vector = np.zeros((2,) + image.shape)
    eigen_vector[0] = hessian_elements[1]
    eigen_vector[1] = hessian_eigenvalues[0] - hessian_elements[0]
    # Return Hessian eigenvalues
    return hessian_eigenvalues, eigen_vector


def _divide_nonzero(array1, array2, cval=1e-10):
    # Copy denominator
    denominator = np.copy(array2)
    # Set zero entries of denominator to small value
    denominator[denominator == 0] = cval
    # Return quotient
    return np.divide(array1, denominator)


def zero_image_border(image):
    height, width = image.shape
    image[0:30, :] = np.zeros((30, width))
    image[height - 30:height, :] = np.zeros((30, width))
    image[:, 0:30] = np.zeros((height, 30))
    image[:, width - 30:width] = np.zeros((height, 30))
    return image


def edge_filter(image, sigmas=range(1, 6, 1)):
    sigmas = np.asarray(sigmas)
    beta = 0.5
    gamma = 15
    beta_sq = 2 * beta ** 2
    gamma_sq = 2 * gamma ** 2

    image = invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros(sigmas.shape + image.shape)
    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):
        # Calculate (abs sorted) eigenvalues
        lambdas, image_direction = compute_hessian_eigenvalues(image, sigma)
        # Compute sensitivity to deviation from a blob-like structure,
        # see equations (10) and (15) in reference [1]_,
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        lambda1, *lambda2 = lambdas
        filtered_raw = np.abs(np.multiply.reduce(lambda2)) ** (1 / len(lambda2))
        r_b = _divide_nonzero(lambda1, filtered_raw) ** 2

        # Compute sensitivity to areas of high variance/texture/structure,
        # see equation (12)in reference [1]_
        r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambda2])

        # Compute output image for given (sigma) scale and store results in
        # (n+1)D matrices, see equations (13) and (15) in reference [1]_
        filtered_array[i] = (np.exp(-r_b / beta_sq) * (1 - np.exp(-r_g / gamma_sq)))
        filtered_array[i] = zero_image_border(filtered_array[i])
        lambdas_array[i] = np.max(lambda2, axis=0)

    # Remove background
    filtered_array[lambdas_array > 0] = 0
    filter_image = np.max(filtered_array, axis=0)
    filter_image = zero_image_border(filter_image)
    filter_image = gaussian(filter_image)
    high_threshold = np.percentile(filter_image, 99.5)
    low_threshold = np.percentile(filter_image, 90)
    edge_image = apply_hysteresis_threshold(filter_image, low_threshold, high_threshold)
    return filter_image, edge_image


x_neighbor = [1, 1, 0, -1, -1, -1, 0, 1]
y_neighbor = [0, 1, 1, 1, 0, -1, -1, -1]


def remove_small_branches(image, RIDGE_THRESHOLD):
    height, width = image.shape

    ending = []
    for y in range(height):
        for x in range(width):
            if image[y][x] == 1:
                count = 0
                for i in range(8):
                    yy = y + y_neighbor[i]
                    if yy < 0 or yy == height: continue
                    xx = x + x_neighbor[i]
                    if xx < 0 or xx == width: continue

                    if image[yy][xx] == 1:
                        count = count + 1

                if count == 1:
                    ending.append([y, x])

    for y, x in ending:
        pixels = []
        ridge_count = 0
        while True:
            count = 0
            for i in range(8):
                yy = y + y_neighbor[i]
                if yy < 0 or yy == height: continue
                xx = x + x_neighbor[i]
                if xx < 0 or xx == width: continue

                if image[yy][xx] == 1:
                    pixels.append([yy, xx])
                    count = count + 1

            if count > 1 or count == 0:
                if ridge_count > RIDGE_THRESHOLD:
                    for yy, xx in pixels:
                        image[yy][xx] = 1
                else:
                    if count == 0 or (abs(pixels[ridge_count][1] - pixels[ridge_count + 1][1]) <= 1 and abs(
                            pixels[ridge_count][0] - pixels[ridge_count + 1][0]) <= 1):
                        image[y][x] = 0
                break
            else:
                image[y][x] = 0
                x = pixels[ridge_count][1]
                y = pixels[ridge_count][0]
                ridge_count = ridge_count + 1


def image_segmentation(image):
    filter_image, edge_image = edge_filter(image)
    res = remove_small_objects(edge_image, min_size=2000)
    res = remove_small_holes(res)
    skel_image = skeletonize(res)
    '''bytimg = img_as_ubyte(skel_image)
    img = Image.fromarray(bytimg)
    img.save("origin.bmp")'''

    remove_small_branches(skel_image, 40)
    '''bytimg = img_as_ubyte(skel_image)
    img = Image.fromarray(bytimg)
    img.save("remove1.bmp")'''
    remove_small_branches(skel_image, 20)
    '''bytimg = img_as_ubyte(skel_image)
    img = Image.fromarray(bytimg)
    img.save("remove2.bmp")'''
    return skel_image, filter_image


def get_vessel_points(image, x0, y0):
    height, width = image.shape
    half_window = 15
    sx = 0
    sy = 0
    s_min = 500
    for y in range(-half_window, half_window + 1):
        for x in range(-half_window, half_window + 1):
            xx = x0 + x
            yy = y0 + y
            rad = x * x + y * y
            if 0 <= xx < width and 0 <= yy < height:
                if image[yy][xx] == 1:
                    if s_min > rad:
                        s_min = rad
                        sx = xx
                        sy = yy

    return s_min, sx, sy


def restore_node(image, node_points):
    for x, y in node_points:
        image[y][x] = 1


final_path = []
cur_path = []


def find_path(image, width, height, x0, y0, x1, y1):
    node_points = []
    global final_path
    global cur_path
    x = x0
    y = y0
    while True:
        count = 0
        neighbor_points = []
        for i in range(8):
            yy = y + y_neighbor[i]
            if yy < 0 or yy == height: continue
            xx = x + x_neighbor[i]
            if xx < 0 or xx == width: continue

            if image[yy][xx] == 1:
                image[yy][xx] = 0
                count = count + 1
                neighbor_points.append([xx, yy])

        res = False
        for i in range(count):
            if x1 == neighbor_points[i][0] and y1 == neighbor_points[i][1]:
                if len(cur_path) < len(final_path) or len(final_path) == 0:
                    final_path = copy.deepcopy(cur_path)
                res = True
                for j in range(count):
                    image[neighbor_points[j][1]][neighbor_points[j][0]] = True
                break
        if res:
            break

        if count == 1:
            node_points.append(neighbor_points[0])
            cur_path.append(neighbor_points[0])
            x = neighbor_points[0][0]
            y = neighbor_points[0][1]

        if count == 0:
            break

        if count > 1:
            for i in range(count):
                cur_path.append(neighbor_points[i])
                find_path(image, width, height, neighbor_points[i][0], neighbor_points[i][1], x1, y1)
                cur_path.remove(neighbor_points[i])

            for x, y in neighbor_points:
                image[y][x] = 1
            break

    restore_node(image, node_points)
    for x in node_points:
        cur_path.remove(x)


def get_initial_path(image, x0, y0, x1, y1):
    global cur_path
    global final_path
    cur_path = []
    final_path = []
    height, width = image.shape
    find_path(image, width, height, x0, y0, x1, y1)
    return final_path


def get_candidate_point(filter_image, template_images, x, y):
    neighbor_pts = [[0, 0], [-16, 0], [16, 0], [0, -16], [0, 16]]
    candidate_pts = []
    for pt, image in zip(neighbor_pts, template_images):
        res = match_template(filter_image, image)
        sy = y - 32 + pt[1]
        ey = y + pt[1]
        sx = x - 32 + pt[0]
        ex = x + pt[0]
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        max_val = np.argmax(res[sy:ey, sx:ex])
        ij = np.unravel_index(max_val, (ey-sy, ex-sx))
        xx, yy = ij[::-1]
        candidate_pts.append([xx + x - 16, yy + y - 16])
    x_mean = 0
    y_mean = 0
    for pt in candidate_pts:
        x_mean += pt[0]
        y_mean += pt[1]
    x_mean //= 5
    y_mean //= 5
    sq_var = np.zeros(5)
    for i in range(5):
        sq_var[i] = (x_mean - candidate_pts[i][0]) * (x_mean - candidate_pts[i][0])\
                    + (y_mean - candidate_pts[i][1]) * (y_mean - candidate_pts[i][1])
    x_mean = 0
    y_mean = 0
    sorted_id = np.argsort(sq_var)

    for i in range(3):
        x_mean += candidate_pts[sorted_id[i]][0]
        y_mean += candidate_pts[sorted_id[i]][1]
    x_mean //= 3
    y_mean //= 3
    return x_mean, y_mean

