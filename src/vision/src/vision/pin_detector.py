import cv2
import datetime
import numpy as np


def pin_tips(image, limits_x, limits_y, cam, plot=True, vertical_shift_cam=0):
    # Convert image to correct color space
    image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    # print("image_cvt.shape: " + str(image_cvt.shape))

    # Convert image to grayscale
    gray = cv2.cvtColor(image_cvt, cv2.COLOR_RGBA2GRAY)

    # Apply a brightness threshold to create a binary image
    thresh = 127  # 200
    binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # print("binary.shape: " + str(binary.shape))

    # Apply pin tip detection kernel to the binary image and erode
    ddepth = -1
    kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
                       [0, 1, 2, 3, 2, 1, 0],
                       [1, 2, 3, 4, 3, 2, 1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [-19, -19, -19, -19, -19, -19, -19],
                       [-19, -19, -19, -19, -19, -19, -19],
                       [-19, -19, -19, -19, -19, -19, -19]])
    result = cv2.filter2D(binary, ddepth, kernel)
    kernel2 = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel2)
    # print("result.shape: " + str(result.shape))

    # Find small contours and their centres

    keypoints = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    max_area = 10  # 5
    white_dots = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < max_area:
            # M = cv2.moments(c)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            cx = 0
            cy = 0
            for p in c:
                cx += p[0][0]
                cy += p[0][1]
            cx = int(cx / len(c))
            cy = int(cy / len(c))
            if limits_x[0] < cx < limits_x[1] and limits_y[0] < cy < limits_y[1]:
                cv2.drawContours(keypoints, [c], -1, (36, 255, 12), 1)
                cv2.circle(keypoints, (cx, cy), 2, (0, 0, 255), -1)
                white_dots.append((cx, cy))

    # print(len(white_dots))
    keypoints = draw_roi_axis(keypoints, limits_x, limits_y, cam, vertical_shift_cam=0)
    # The result is overlayed with the analyzed image for validation
    a_channel = np.ones(image_cvt.shape, dtype=float) / 1.7
    overlay = image_cvt * a_channel
    alpha = cv2.split(overlay)[3]
    mask = (np.multiply(alpha, 1.0 / 255))[:, :, np.newaxis]
    output = cv2.convertScaleAbs(image_cvt * mask + cv2.cvtColor(keypoints, cv2.COLOR_BGR2BGRA))  # * (1 - mask)
    output = draw_roi_axis(output, limits_x, limits_y, cam, vertical_shift_cam)
    if plot:
        cv2.imshow('Pin tip detection', output)
        # now = datetime.datetime.now()
        # now_str = now.strftime('%Y-%m-%dT%H%M%S-%f')
        # cv2.imwrite('/home/ips/Pictures/230725_pin_detection_images/'+now_str+'.png', output)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    return white_dots

    # For visualization purposes: overlay detected 'pin tips' on the source image
    # a_channel = np.ones(image_cvt.shape, dtype=float) / 2.0
    # overlay = image_cvt * a_channel
    # alpha = cv2.split(overlay)[3]
    # mask = (np.multiply(alpha, 1.0 / 255))[:, :, np.newaxis]
    # output = cv2.convertScaleAbs(image_cvt * mask + cv2.cvtColor(result, cv2.COLOR_GRAY2RGBA) * (1 - mask))
    # output = cv2.convertScaleAbs(image_cvt * mask + result * (1 - mask))
    # return output


def draw_roi_axis(image, limits_x, limits_y, cam, vertical_shift_cam):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.rectangle(image, (limits_x[0], limits_y[0]), (limits_x[1], limits_y[1]), (0, 255, 0))
    rows, cols, _ = image.shape
    axis_length = 45
    o_x1 = int(cols * 0.9) # 810
    o_x2 = int(cols * 0.9) - axis_length  # 765
    o_y = int(rows * 0.9)  # 450

    if cam == 1:
        image = cv2.arrowedLine(image, (o_x1, o_y), (o_x1 - axis_length, o_y), (0, 0, 255), 2)
        image = cv2.arrowedLine(image, (o_x1, o_y), (o_x1, o_y - axis_length), (0, 0, 255), 2)
        image = cv2.putText(image, 'y_tilda', (o_x1 - axis_length - 70, o_y), font, 0.65, (0, 0, 255))
        image = cv2.putText(image, 'z_tilda', (o_x1 - axis_length, o_y - axis_length - 10), font, 0.65, (0, 0, 255))
    elif cam == 2:
        image = cv2.arrowedLine(image, (o_x2, o_y), (o_x2 + axis_length, o_y), (0, 0, 255), 2)
        image = cv2.arrowedLine(image, (o_x2, o_y), (o_x2, o_y - axis_length), (0, 0, 255), 2)
        image = cv2.putText(image, 'x_tilda', (o_x2 + axis_length + 10, o_y), font, 0.65, (0, 0, 255))
        image = cv2.putText(image, 'z_tilda', (o_x2 - 25, o_y - axis_length - 10), font, 0.65, (0, 0, 255))

    # print rows/2, cols/2
    # Vertical axis of crosshairs
    image = cv2.line(image, (int(cols / 2), 0), (int(cols / 2), int(rows)), (255, 0, 0), 1)
    # Horizontal axis of crosshairs
    if vertical_shift_cam is not None:
        print('vertical_shift_cam: ', vertical_shift_cam)
        color_blue = (255, 0, 0)
        color_yellow = (0, 255, 255)
        if vertical_shift_cam == 0:
            # Ideal horizontal line
            image = cv2.line(image, (0, int(rows / 2)),
                             (int(cols), int(rows / 2)), color_blue, 1)
        else:
            # Corrected horizontal line
            image = cv2.line(image, (0, int(rows / 2) - vertical_shift_cam),
                             (int(cols), int(rows / 2) - vertical_shift_cam), color_yellow, 1)

    return image


def get_pin_tip_location(tip_coords_1, tip_coords_2, x_tilda_c1, y_tilda_c2, B, H, L):
    x1 = tip_coords_1[0]
    y1 = tip_coords_1[1]
    x2 = tip_coords_2[0]
    y2 = tip_coords_2[1]
    a = 0.0
    b = 0.0
    pixel_tolerance = 8
    # print('Getting pin tip coordinates based on pixel inputs:')
    # print('x1 = %d, x2 = %d, y1 = %d, y2 = %d' % (x1, x2, y1, y2))
    if x1 == int(B / 2):
        a = -x_tilda_c1 / L
        b = -y_tilda_c2 / L
        if x2 == int(B / 2):
            # point is on the z-tilda axis
            if y1 == int(H / 2):
                if y2 == int(H / 2):
                    # point is in the origin of tilda coordinate system
                    return 0.0, 0.0, 0.0
                elif y2 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance):
                    print('Warning! Optical axis of cameras do not cross! 1')
                    return 0.0, 0.0, 0.0
                else:
                    print('ERROR! Optical axis of cameras do not cross! 1')
            else:
                if y1 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('Warning! Optical axis of cameras do not cross! 2')
                    return 0.0, 0.0, 0.0
                elif y1 not in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('ERROR! Optical axis of cameras do not cross! 2')
                else:
                    # point is on the z-tilda axis but not in the origin
                    # a = -x_tilda_c1 / L
                    # b = -y_tilda_c2 / L
                    return 0.0, 0.0, a * (int(H / 2) - y1)
        else:
            # point is on the xz-tilda plane
            # b = -y_tilda_c2 / L
            if y1 == int(H / 2):
                if y2 == int(H / 2):
                    # point is on the x-tilda axis
                    return b * (x2 - int(B / 2)), 0.0, 0.0
                elif y2 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance):
                    print('Warning! Optical axis of cameras do not cross! 3')
                    return b * (x2 - int(B / 2)), 0.0, 0.0
                else:
                    print('ERROR! Optical axis of cameras do not cross! 3')
            else:
                if y1 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('Warning! Optical axis of cameras do not cross! 4')
                    return b * (x2 - int(B / 2)), 0.0, 0.0
                elif y1 not in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('ERROR! Optical axis of cameras do not cross! 4')
                else:
                    # point is on the xz-tilda plane but not on the x-tilda axis
                    return b * (x2 - int(B / 2)), 0.0, b * (y2 - int(H / 2))
    else:
        if x2 == int(B / 2):
            # point is on the yz-tilda plane
            a = -x_tilda_c1 / L
            if y1 == int(H / 2):
                if y2 == int(H / 2):
                    # point is on the y-tilda axis
                    return 0.0, a * (int(B / 2) - x1), 0.0
                elif y2 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance):
                    print('Warning! Optical axis of cameras do not cross! 5')
                    return 0.0, a * (int(B / 2) - x1), 0.0
                else:
                    print('ERROR! Optical axis of cameras do not cross! 5')
            else:
                if y1 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('Warning! Optical axis of cameras do not cross! 6')
                    return 0.0, a * (int(B / 2) - x1), 0.0
                elif y1 not in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('ERROR! Optical axis of cameras do not cross! 6')
                else:
                    # point is on the yz-tilda plane but not on the y-tilda axis
                    return 0.0, a * (int(B / 2) - x1), a * (int(H / 2) - y1)
        else:
            # point is not on the xz-tilda and not on the yz-tilda planes
            b = (x_tilda_c1 * (int(B / 2) - x1) + y_tilda_c2 * L) / ((x2 - int(B / 2)) * (int(B / 2) - x1) - L * L)
            a = (y_tilda_c2 + b * L) / (int(B / 2) - x1)
            if y1 == int(H / 2):
                if y2 == int(H / 2):
                    # point is on the xy-tilda plane but not on the x-tilda or y-tilda axes
                    return b * (x2 - int(B / 2)), a * (int(B / 2) - x1), 0.0
                elif y2 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance):
                    print('Warning! Optical axis of cameras do not cross! 7')
                    return b * (x2 - int(B / 2)), a * (int(B / 2) - x1), 0.0
                else:
                    print('ERROR! Optical axis of cameras do not cross! 7')
            else:
                # point is also not on xy-tilda plane
                if y1 in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('Warning! Optical axis of cameras do not cross! 8')
                    return b * (x2 - int(B / 2)), a * (int(B / 2) - x1), a * (int(H / 2) - y1)
                elif y1 not in range(int(H / 2) - pixel_tolerance, int(H / 2) + pixel_tolerance) and y2 == int(H / 2):
                    print('ERROR! Optical axis of cameras do not cross! 8')
                else:
                    # general case (point is not on the planes or the axes)
                    return b * (x2 - int(B / 2)), a * (int(B / 2) - x1), a * (int(H / 2) - y1)
