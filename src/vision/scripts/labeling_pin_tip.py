

from vision.labeling_click_goal import LabelingGoalPixelSetter
from vision.extract_expert_data import get_images_from_expert_data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

import matplotlib.pyplot as plt

class ImageIter:
    def __init__(self, path):
        self.path = path
        images1, images2 = get_images_from_expert_data(path)
        self.images1 = images1
        self.images2 = images2
        self.counter = 0
        self.l = len(images1)
    def isOpened(self):
        return self.counter < self.l

    def release(self):
        pass

    def destroyAllWindows(self):
        pass

    def read(self):
        img = self.images1[self.counter]
        img = img/255.0
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        copy_img = img.astype(np.float32).copy()
        #img = cv2.bilateralFilter(img.astype(np.float32), 3, 35, 35)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = np.asarray(img, np.float32)
        #img = img[25:35,20:30]
        #img = np.mean(img, axis=2)
        print(img)
        #img = cv2.resize(img, None, fx=2.0, fy=2.0)
        #img = cv2.pyrUp(img, (400,400))
        # Sharpen the image
        #img = cv2.filter2D(img, -1, kernel)
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
        #kernel = np.ones((3, 3), np.float32) / 9
        #kernel = 18* (0.5-np.random.rand(3,3)) / 9

        #print(kernel)
        #img = cv2.filter2D(img, -1, kernel)
        #kernel = 18 * (0.5 - np.random.rand(3, 3)) / 9

        #print(kernel)
        #img = cv2.filter2D(img, -1, kernel)
        #kernel = 18 * (0.5 - np.random.rand(3, 3)) / 9

        #print(kernel)
        #img = cv2.filter2D(img, -1, kernel)
        img = 1 - img
        Conv_hsv_Gray = np.mean(img, axis=2)
        indices = np.where(Conv_hsv_Gray<=0.98)
        img[indices[0], indices[1], :] = [0.0, 0.0, 0.7]
        img = cv2.addWeighted(np.asarray(copy_img, np.float64),.85, np.asarray(img, np.float64), 0.1, 0)
        plt.imshow(copy_img)
        plt.show()
        #print(img)
        self.counter = self.counter + 1
        return True, img

if __name__ == "__main__":
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H%M%S")

    save_folder = 'C:\\Users\\boiar\\Documents\\Data\\SmartAssembly\\imgs_training'
    save_folder = save_folder + '\\' + date_time
    isExist = os.path.exists(save_folder)
    if not isExist:
        # Create a new directory because it does not exist
        print("folder " + save_folder + " is created")
        os.makedirs(save_folder)

    videofile = 'C:\\Users\\boiar\\Documents\\Data\\SmartAssembly\\cam1_pin.avi'
    #cap = cv2.VideoCapture(videofile)
    cap = ImageIter("C:\\Users\\boiar\\Documents\\Gitlab\\SmartAssembly\\ips-smartassembly\\expertdata\\session_5\\2023-03-13-14-53-14_expert_0.1.npz")
    # Setting goal hole by clicking it in both camera images
    print("THTRLEnv: Please set the goal hole by clicking on it in both images!")
    goal_pixel_setter = LabelingGoalPixelSetter()
    counter = 0
    with open(save_folder + '\\labeling.txt', "a+") as f: # create if not exists
        try:
            while(cap.isOpened()):
                counter = counter + 1
                print("counter",counter)
                ret, frame = cap.read()
                if ret == False:
                    print("Video stream is not delivering an image. Try again!")
                    break
                else:
                    #if not (counter % 15 == 0): # skip 14 images
                    #    continue
                    #if counter < 15*2*60:
                    #    continue
                    goal_pixel_setter = LabelingGoalPixelSetter()
                    goal_pixel_setter.click_set_goal_in_pixels(frame)
                    if goal_pixel_setter.done:
                        print("Aborted by right clicking")
                        break
                    else:
                        if goal_pixel_setter.x is None or goal_pixel_setter.y is None:
                            print("Not clicked in image. Try again!")
                            time.sleep(1)
                        else:
                            goal_pose_pixel_image = (int(goal_pixel_setter.x), int(goal_pixel_setter.y))
                            print("Clicked goal pixel in image: ", goal_pose_pixel_image)
                            name = date_time + 'img_' + str(counter)
                            f.write(name + ";" + str(goal_pose_pixel_image) + ";" + videofile +"\n")
                            frame = frame * 255
                            cv2.imwrite(save_folder + '\\' +name + '.png', frame)
        except KeyboardInterrupt:
            print('Stopped by keyboard interrupt')
    cap.release()
    cv2.destroyAllWindows()
