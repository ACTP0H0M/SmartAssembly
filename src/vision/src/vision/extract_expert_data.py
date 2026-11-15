
import numpy as np
import glob
import os
import cv2
def get_images_from_expert_data(expert_data, swapaxis = True):
    """
        Load expert data with np.load(expert_data), reshape np to shape
        (batches, channels, width, height) = (batches, 3, 40, 40)
        @param expert_data: path to expert data
        @return: image1, image2
    """
    transitions = np.load(expert_data)

    obs = transitions["obs"]  # transform to dict and reshape images
    print('Observation shape: ', obs.shape)

    image1 = obs[:, :40 * 40 * 3]
    image1 = image1.reshape((obs.shape[0], 40, 40, 3))
    if swapaxis:
        image1 = np.swapaxes(image1, 2, 3)
        image1 = np.swapaxes(image1, 1, 2)

    image2 = obs[:, 40 * 40 * 3:2 * 40 * 40 * 3]
    image2 = image2.reshape((obs.shape[0], 40, 40, 3))
    if swapaxis:
        image2 = np.swapaxes(image2, 2, 3)
        image2 = np.swapaxes(image2, 1, 2)

    return image1, image2


if __name__ == "__main__":
    path = 'C:\\Users\\boiar\\Documents\\Data\\SmartAssembly\\'
    path_expert_data = path+'expertdata\\'
    save_folder = path+'expertdata_png\\'
    train_sessions = [6, 7, 8]
    if not os.path.isdir(path_expert_data):
        raise IOError("folder ", path_expert_data, " not exists! Please create this empty folder!")
    for train_session in train_sessions:
        if not os.path.isdir(path_expert_data):
            raise IOError("folder ", save_folder + "session_" + str(train_session), " not exists! Please create this empty folder!")
    filenames = [f for f in glob.glob(path_expert_data + '\\**\\*.npz', recursive=True)]
    filenames = [f for f in filenames if any(
        "session_" + str(session) in f for session in train_sessions)]  # TODO session 1 is not selecting session 10
    print(filenames)
    loaded_transitions = [get_images_from_expert_data(path_expert_data_and_filename,swapaxis=False) for path_expert_data_and_filename in filenames]
    for name, (img1,img2) in zip(filenames, loaded_transitions):
        print(name, img1.shape,img2.shape)
        id = "\\".join(name.split("\\")[-2:])
        assert (img1.shape[0] == img2.shape[0])
        for batch_count in range(img1.shape[0]):
            cv2.imwrite(save_folder + id + '_cam1_'+ str(batch_count) +'.png', img1[batch_count])
            cv2.imwrite(save_folder + id + '_cam2_'+ str(batch_count) +'.png', img2[batch_count])

