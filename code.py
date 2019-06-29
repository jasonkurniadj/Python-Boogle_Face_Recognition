import os
import cv2
import math
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

    ret_subdirectories = []
    train_subfolder = os.listdir(root_path)

    for subfolder in enumerate(train_subfolder):
        ret_subdirectories.append(subfolder)

    return ret_subdirectories

def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''

    ret_images_path = []
    ret_classes_id = []

    for i, subfolder in train_names:
        full_path = root_path + '/' + subfolder
        image_list = os.listdir(full_path)

        for filename in image_list:
            img_path = full_path + '/' + filename
            print(img_path)

            ret_images_path.append(img_path)
            ret_classes_id.append(i)

    return ret_images_path, ret_classes_id

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

    print("Get train images...")
    ret_train_images = []

    for path in image_path_list:
        ret_train_images.append(cv2.imread(path))

    return ret_train_images

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

    print("Detect faces:")

    ret_images_gray = []
    ret_rect_position = []
    ret_classes_id = []

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for i, img in enumerate(image_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.25, minNeighbors=5)

        for x,y,w,h in detected_faces:
            print(i,":", x,y,w,h)
            curr_face = img_gray[y:y+h, x:x+w]

            ret_images_gray.append(curr_face)
            ret_rect_position.append((x, y, w, h))

            if image_classes_list != None:
                ret_classes_id.append(image_classes_list[i])
    
    return ret_images_gray, ret_rect_position, ret_classes_id

def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''

    print("Train images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer

def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''

    print("Get test images...")
    ret_test_images = []

    for _, filename in image_path_list:
        full_path = test_root_path + '/' + filename
        ret_test_images.append(cv2.imread(full_path))

    return ret_test_images

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    print("Predict images:")
    ret_prediction_res = []

    for img in test_faces_gray:
        idx, confidence = classifier.predict(img)
        confidence = math.floor(confidence * 100) / 100

        print((idx, confidence))
        ret_prediction_res.append((idx, confidence))

    return ret_prediction_res

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''

    print("Draw results...")
    ret_final_images = []
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    for i, img in enumerate(test_image_list):
        x = test_faces_rects[i][0]
        y = test_faces_rects[i][1]
        s = test_faces_rects[i][2]

        idx = predict_results[i][0]
        text = str(train_names[idx][1])

        cv2.rectangle(img, (x,y), (x+s, y+s), (0,255,0), 1)
        cv2.putText(img, text, (x,y-1), font_face, 0.5, (0,255,0))

        ret_final_images.append(img)

    return ret_final_images

def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''

    ret_images_arr = np.array(predicted_test_image_list)
    return ret_images_arr

def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

    horizontal_image = np.hstack(image)
    print("FINISH!!!")

    cv2.imshow('Boogle Result', horizontal_image)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train_temp"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)
    
    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)