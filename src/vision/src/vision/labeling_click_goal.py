# importing the module
import cv2

class LabelingGoalPixelSetter():
    def __init__(self):
        self.x = None
        self.y = None
        self.img = None
        self.done = False

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks, right click opens drop down menu
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imshow("Window_name", self.img)
            self.x = x
            self.y = y
            print("clicked: x=",x," y=",y)
            cv2.destroyAllWindows()
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.imshow("Window_name", self.img)
            self.done = True
            cv2.destroyAllWindows()



    # driver function
    def click_set_goal_in_pixels(self, img):
        # displaying the image
        self.img = img
        cv2.namedWindow("Window_name", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window_name", 100, 100)
        #cv2.setWindowProperty("Window_name", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Window_name", self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback("Window_name", self.click_event)
        # wait for a key to be pressed to exit
        cv2.waitKey(10000)
        # close the window
        cv2.destroyAllWindows()