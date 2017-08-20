import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.stats
from skimage.feature import peak_local_max

from skimage.morphology import watershed, disk
from skimage.filters import rank

def play_movie(movie):
    t = movie.shape[-1]

    frame_counter = 0
    while True:
        cv2.imshow('Movie', cv2.resize(movie[:, :, frame_counter]/255, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        frame_counter += 1
        if frame_counter == t:
            frame_counter = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def normalize(image, force=False):
    min_val = np.amin(image)

    image_new = image.copy()

    if min_val < 0:
        image_new -= min_val

    max_val = np.amax(image_new)

    if force:
        return 255*image_new/max_val

    if max_val <= 1:
        return 255*image_new
    elif max_val <= 255:
        return image_new
    else:
        return 255*image_new/max_val

def order_statistic(image, percentile, window_size):
    order = int(np.floor(percentile*window_size**2))
    return ndi.rank_filter(image, order, size=(window_size, window_size))

def rescale_0_1(image):
    (h, w) = image.shape
    n_pixels = h*w
    S = np.reshape(image, (1, n_pixels))

    percent = 5.0/10000
    min_percentile = int(np.maximum(np.floor(n_pixels*percent), 1))
    max_percentile = n_pixels - min_percentile

    S = np.sort(S)

    denominator = S[0, max_percentile] - S[0, min_percentile]

    if abs(denominator) == 0:
        denominator = 1e-6

    return (image - S[0, min_percentile])/denominator

def calculate_local_correlations(movie):
    (h, w, t) = movie.shape

    cross_correlations = np.zeros((h, w, 2, 2))

    index_range = [-1, 1]

    for i in range(h):
        for j in range(w):
            for k in range(2):
                for l in range(2):
                    ind_1 = index_range[k]
                    ind_2 = index_range[l]
                    cross_correlations[i, j, ind_1, ind_2] = scipy.stats.pearsonr(movie[i, j], movie[min(max(i+ind_1, 0), h-1), min(max(j+ind_2, 0), w-1)])[0]
    
    correlations = np.mean(np.mean(cross_correlations, axis=-1), axis=-1)

    return correlations

def mean(movie):
    return np.mean(movie, axis=-1)

def adjust_contrast(image, contrast):
    return np.minimum(contrast*image, 255)

def adjust_gamma(image, gamma):
    new_image = image/255.0
    new_image = new_image**(1.0/gamma)
    return np.minimum(255*new_image, 255)

def apply_watershed(original_image, cells_mask, starting_image, soma_threshold, compactness, centers_list):
    if len(original_image.shape) == 2:
        rgb_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = original_image.copy()

    (h, w) = starting_image.shape

    threshold_image = starting_image.copy()

    # threshold_image[threshold_image < soma_threshold] = 0
    # threshold_image[threshold_image > 0] = 255
    threshold_image[cells_mask == 0] = 0

    # rgb_image = cv2.cvtColor(threshold_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    D = ndi.distance_transform_edt(threshold_image)

    localMax = peak_local_max(D, indices=False, min_distance=0, labels=threshold_image)

    L = D.astype(np.uint8)
    _, markers = cv2.connectedComponents(L)

    labels = watershed(L, markers.astype(np.int32), watershed_line=True, compactness=compactness)
    # labels = cv2.watershed(cv2.cvtColor(L, cv2.COLOR_GRAY2RGB), markers.astype(np.int32))

    for label in np.unique(labels):
        if label <= 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(threshold_image.shape, dtype="uint8")
        mask[labels == label] = 255
        # mask[label > 0] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        M = cv2.moments(c)
        area = cv2.contourArea(c)

        if len(c) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(c)

            if centers_list != []:
                min_distance = min([ np.sqrt((x - pos[0])**2 + (y - pos[1])**2) for pos in centers_list ])

            n_pixels = len(c)
            region_size = np.sqrt(n_pixels)

            # print(len(c[0][0]))
            mean_dispersion = np.sqrt(np.mean([ (c[i][0][0] - x)**2 + (c[i][0][1] - y)**2 for i in range(n_pixels) ]))/region_size

            # mean_intensity = np.mean([ starting_image[k, l] for k in np.arange(max(0, int(x) - 5), min(h, int(x) + 6)) for l in np.arange(max(0, int(y) - 5), min(w, int(y) + 6)) ])
            # print(mean_intensity)

            if M["m00"] != 0 and (7 < area < 100 and 0.2 < MA/ma < 1.9) and (centers_list == [] or min_distance >= 3) and mean_dispersion <= 1:
            # if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                color = np.random.uniform(0.2, 255, size=(3,))
                cv2.circle(rgb_image, (int(x), int(y)), 1, color, -1)

                overlay = rgb_image.copy()
                overlay[mask > 0] = color.astype(np.uint8)

                # cells_mask[labels == label] = 0

                cv2.addWeighted(overlay, 0.4, rgb_image, 0.6, 0, rgb_image)

                centers_list.append([x, y])

    #     cv2.imshow("Contours", cv2.resize(rgb_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    #     cv2.waitKey(1)
    # cv2.waitKey(0)

    return rgb_image, cells_mask, centers_list