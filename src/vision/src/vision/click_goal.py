# importing the module
import cv2

class GoalPixelSetter:
    def __init__(self):
        self.x = None
        self.y = None
        self.img = None

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks, right click opens drop down menu
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imshow('image', self.img)
            self.x = x
            self.y = y



    # driver function
    def click_set_goal_in_pixels(self, img):
        # displaying the image
        self.img = img
        cv2.imshow('image', self.img)
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)
        # wait for a key to be pressed to exit
        cv2.waitKey()
        # close the window
        cv2.destroyAllWindows()