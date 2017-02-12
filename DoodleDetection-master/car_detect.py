# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# opencv uses BGR
import sys

wheel_color = (171, 186, 211)[::-1]
body_color = (255, 0, 0)[::-1]


def contour_center(contour):
    M2 = cv2.moments(contour)
    return (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))


def resize_contour(contour, factor):
    M = cv2.moments(contour)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    scaled_contour = np.copy(contour)

    for i in range(scaled_contour.shape[0]):
        scaled_contour[i][0][0] = int(contour[i][0][0] * factor)
        scaled_contour[i][0][1] = int(contour[i][0][1] * factor)

    M2 = cv2.moments(scaled_contour)
    new_center = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

    # maintains center
    for i in range(scaled_contour.shape[0]):
        scaled_contour[i][0][0] -= new_center[0] - center[0]
        scaled_contour[i][0][1] -= new_center[1] - center[1]

    return scaled_contour


def add_alpha(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255  # creating a dummy alpha channel image.
    img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_RGBA


def fill(img, contour, color):
    corners = np.zeros((1, contour.shape[0], 2), dtype=np.int32)
    for i in range(contour.shape[0]):
        corners[0][i][0] = contour[i][0][0]
        corners[0][i][1] = contour[i][0][1]
    cv2.fillPoly(img, corners, color)


def crop(image, contour, output_fname):
    # paint wheel
    center = contour_center(contour)
    flood_fill(image, center, wheel_color)

    image = add_alpha(image)
    mask = np.zeros(image.shape, dtype=np.uint8)

    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    fill(mask, contour, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite(output_fname, masked_image)
    with open(output_fname + '_.info', 'w') as info:
        info.write('{}x{}'.format(center[0], center[1]))

    return masked_image, center


def flood_fill(img, seed, color):
    blank_image = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, blank_image, seed, color)


def extract_wheels(image, circles, output_index=0):
    image = image.copy()
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[:]).astype("int")
    contour_list = []
    radius = max(int(circles[0][2] * 2), int(circles[1][2] * 1.3))
    cropped_images = []
    centers = []
    approxims = []
    wheels = []
    wheel_offsets = []
    # loop over the (x, y) coordinates and radius of the circles
    for j, (x, y, r) in enumerate(circles):
        # crop image
        cropped_image = image[y - radius:y + radius, x - radius:x + radius]
        wheel_offsets.append((y - radius, x - radius))
        cropped_images.append(cropped_image)
        bilateral_filtered_image = cv2.bilateralFilter(cropped_image, 5, 175, 175)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
        _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("output", cropped_image)
        # cv2.waitKey(0)

        # just take contour with greatest area
        max_approx = (None, -1)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(approx)

            if area > max_approx[1]:
                max_approx = (approx, area)

        approx = max_approx[0]
        approx = resize_contour(approx, 1.1)
        contour_list.append(approx)
        wheels.append(crop(cropped_image, approx, 'wheel_{}_{}.png'.format(j, output_index)))

        for i in range(approx.shape[0]):
            approx[i, 0, 0] += x - radius
            approx[i, 0, 1] += y - radius
        # fill(image, approx, (255, 255, 255))
        center = contour_center(approx)
        centers.append(center)

        approxims.append(approx)

    # paint car body
    # estimate body center position based on wheel proportions
    x_dist = abs(centers[0][0] - centers[1][0])
    x_avg = centers[0][0] / 2 + centers[1][0] / 2
    y_avg = centers[0][1] / 2 + centers[1][1] / 2
    body_center = (x_avg, int(y_avg - x_dist * 0.3))
    flood_fill(image, body_center, body_color)

    # remove wheels from car
    fill(image, approxims[0], (255, 255, 255))
    fill(image, approxims[1], (255, 255, 255))

    cv2.imwrite('car_without_wheels_{}.png'.format(output_index), image)

    to_imgs(image, wheels[0][0], wheels[0][1], wheel_offsets[0], wheels[1][0], wheels[1][1], wheel_offsets[1])

    # make_gif()


def check_circle(c, box):
    # is center of circle in box?
    return box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]


def check_model_output(circles, label):
    if circles is None or len(circles) != 2:
        return False

    return check_circle(circles[0], label[0]) and check_circle(circles[1], label[1]) \
           or check_circle(circles[0], label[1]) and check_circle(circles[1], label[0])


def filter(circles):
    if len(circles) > 2:
        # filter out circle with higher radius
        circles = sorted(circles, key=lambda x: x[2])
        # only return 2 smallest circles
        return circles[:2]
    else:
        return circles


def graph(image, circles):
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[:]).astype("int")
        contour_list = []
        radius = max(int(circles[0][2] * 2), int(circles[1][2] * 1.3))
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # crop image
            cropped_image = image[y - radius:y + radius, x - radius:x + radius]
            bilateral_filtered_image = cv2.bilateralFilter(cropped_image, 5, 175, 175)
            edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
            _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow("output", cropped_image)
            # cv2.waitKey(0)

            # just take contour with greatest area
            max_approx = (None, -1)

            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
                area = cv2.contourArea(approx)

                if area > max_approx[1]:
                    max_approx = (approx, area)

            approx = max_approx[0]
            for i in range(approx.shape[0]):
                approx[i, 0, 0] += x - radius
                approx[i, 0, 1] += y - radius
            contour_list.append(approx)

            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(image, (x, y), radius, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.drawContours(image, contour_list, -1, (255, 0, 0), 2)
        cv2.imshow("output", image)
        cv2.waitKey(0)


def rotateImage(image, center, angle):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, (image.shape[0], image.shape[1]))
    return result


def blend_transparent(background, img_trans):
    # Split out the transparency mask from the colour info
    overlay_img = img_trans[:, :, :3]  # Grab the BRG planes
    overlay_mask = img_trans[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (background * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def shift_left(img, x):
    M = np.float32([[1, 0, -x], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255, 255))


def make_gif():
    import imageio

    images = []
    dir = 'animation'
    for filename in os.listdir(dir):
        images.append(imageio.imread(os.path.join(dir, filename)))
    imageio.mimsave('car.gif', images)


def to_imgs(body, wheel1, wheel1_center, wheel1_offset, wheel2, wheel2_center, wheel2_offset):
    iterations = 25
    for i in range(iterations):
        wheel1_rot = rotateImage(wheel1, wheel1_center, 1000 / iterations * i)
        wheel2_rot = rotateImage(wheel2, wheel2_center, 1000 / iterations * i)

        wheel1_frame = np.zeros((body.shape[0], body.shape[1], 4), dtype=np.uint8)
        wheel1_frame[wheel1_offset[0]:wheel1_offset[0] + wheel1_rot.shape[0],
        wheel1_offset[1]: wheel1_offset[1] + wheel1_rot.shape[1]] = wheel1_rot
        merged = blend_transparent(body, wheel1_frame)

        wheel2_frame = np.zeros((body.shape[0], body.shape[1], 4), dtype=np.uint8)
        wheel2_frame[wheel2_offset[0]:wheel2_offset[0] + wheel2_rot.shape[0],
        wheel2_offset[1]: wheel2_offset[1] + wheel2_rot.shape[1]] = wheel2_rot
        merged = blend_transparent(merged, wheel2_frame)

        shifted = shift_left(merged, 1000 / iterations * i)

        cv2.imwrite('animation/rot_{}.png'.format(i), shifted)


def process_image(image, output_index):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.3, 250, param2=70)

    if circles is None:
        print('No wheels detected!')
        return

    circles = circles[0]

    circles = filter(circles)

    # graph(image, circles)

    if len(circles) < 2:
        print('Fewer than two circles detected!')
        return

    extract_wheels(image, circles, output_index)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('This script takes exactly one argument, which is the input file!')
        exit(1)
    input_fname = sys.argv[1]
    process_image(cv2.imread(input_fname), 0)
