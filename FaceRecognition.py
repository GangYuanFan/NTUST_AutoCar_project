"""
@ Real-Time face recognition with VGG-16 CNN structure
this code can recognize Jerry and N-Way or not.

@ VGG-16
Loss: cross-entropy with softmax and L1 regularity
Optimizer: using Adam-OP
"""
from __future__ import print_function
import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from imutils import face_utils
import dlib
import math
from PIL import ImageEnhance, Image


class Neural_Network(object):
    def __init__(self):
        self.close_eye_counter = 0
        self.driver_ID = "Stranger"
        self.GPUused = 0
        self.CAM_FLAG = 0
        self.image_num = 500  # image number of each class

        self.pretrained_model_dir = 'D:/DeepLearning/face/CNN_model_vgg/pre-trained/'
        self.neg_face_dir = 'D:/DeepLearning/face/trainFace/GFW/'
        self.pos_face_dir = 'D:/DeepLearning/face/trainFace/MyFace/'
        self.classfication = {0: "Mark", 1: "Kevin", 2: "Jerry", 3: "Other_Driver", 4: "Stranger"}

        fname = 'D:/DeepLearning/face/trainFace/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(fname)
        self.test_folder = "D:/DeepLearning/face/testFace/MyFace/"
        self.close_eye_flag = False
        self.counter = 0

        # open dataset dir
        self.data_path = [self.pos_face_dir + f + "/" for f in os.listdir(self.pos_face_dir) if f[0:4] == 'Face']
        self.data_path.append(self.neg_face_dir)
        self.n_class = len(self.data_path)

        # open pre-trained model
        with open(self.pretrained_model_dir + "vggface16.tfmodel", mode='rb') as f:
            fileContent = f.read()
        self.graph_def = tf.GraphDef()
        self.graph_def.ParseFromString(fileContent)
        self.graph = tf.get_default_graph()

        with tf.name_scope('xs'):
            # define placeholder for inputs to network
            self.xs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.RGBmean = [94.84945632378849, 103.65879065577424, 123.42070654283751]
            blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=self.xs)
            self.xs2 = tf.concat(axis=3, values=[
                blue - self.RGBmean[0],
                green - self.RGBmean[1],
                red - self.RGBmean[2],
            ])

        with tf.name_scope('y_label'):
            self.ys = tf.placeholder(tf.float32, [None, self.n_class])

        with tf.name_scope('hyperParameter'):
            self.learning_rate = tf.placeholder(tf.float32)

        self.vgg16()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.clf = joblib.load('C:/Users/Jerry/PycharmProjects/face/' + 'vgg_fc7_svm.pkl')
        print('Model Restored Completely')

    def shuffle(self, x, y):
        """
        :param x: (n_samples, n_inputs)
        :param y: (1, n_samples)
        :return: rearranged x and y
        """
        indices = np.random.permutation(len(y))
        print("random indices created!")
        return x[indices], y[indices]

    def normalize(self, data):
        """
        normalize
        :param data: inputs data
        :return: normalized data
        """
        mean = data.mean()
        stddev = data.std()
        newdata = (data - mean) / stddev
        newdata = newdata - newdata.min()
        data = newdata / newdata.max()
        return data

    def read_data_to_4d(self, images_dir, image_num):
        """
        read images from folder, and shuffling, reshape into 4-D ndarray
        :param images_dir: images folder
        :param image_num: number of images read
        :return: 4-D ndarray images
        """
        images_file = np.array([images_dir + f for f in os.listdir(images_dir) if f[-4:] == ".jpg" or f[-4:] == ".png"])
        indices = np.random.permutation(len(images_file))
        images_file = images_file[indices]

        img_list = []
        if len(images_file) >= image_num:
            for f in images_file[0: image_num]:
                img_list.append(cv2.imread(f, 1))
            img_list = np.array(img_list, dtype=np.float32)
        else:
            raise FileExistsError("Data Not Enough")
        return img_list

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def vgg16(self):
        # using pretrained model
        tf.import_graph_def(self.graph_def, input_map={"images": self.xs2})
        self.h_fc7 = tf.nn.relu(self.graph.get_tensor_by_name("import/fc7/MatMul:0"))

    def data_collection(self, img, IMG_counter):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect, landmark = self.get_detect(gray)
        if len(rect) is not 0:
            area = np.zeros(shape=len(rect))
            img_counter = 0
            for (x, y, w, h) in rect:
                area[img_counter] = w * h
                img_counter += 1
            maxval, idx = self.Maximum(area)
            crop_img = img[rect[idx][1]:rect[idx][1] + rect[idx][3], rect[idx][0]:rect[idx][0] + rect[idx][2], :]
            crop_img = cv2.resize(crop_img, (224, 224))
            cv2.imwrite(self.pos_face_dir + "Face4/MyFace_%d.jpg" % IMG_counter, crop_img)
        return len(rect)

    def data_augmentation(self):
        files = os.listdir(self.pos_face_dir + "Face4/")
        srcIMG_path = self.pos_face_dir + "Face4/"
        dstIMG_path = srcIMG_path
        file_counter = 0
        for f in files:
            if (not f[-4:] == ".jpg" and not f[-4:] == ".png") or f[:6] != "MyFace":
                continue
            img = cv2.imread(srcIMG_path + f, 1)

            # Flipping images with Numpy
            flipped_H_img = np.fliplr(img)
            cv2.imwrite(dstIMG_path + "flipped_H_" + f, flipped_H_img)

            # Flipping images with Numpy
            flipped_V_img = np.flipud(img)
            cv2.imwrite(dstIMG_path + "flipped_V_" + f, flipped_V_img)

            # Flipping images with Numpy
            flipped_HV_img = np.flipud(flipped_H_img)
            cv2.imwrite(dstIMG_path + "flipped_HV_" + f, flipped_HV_img)

            # Gaussian noise
            row, col, ch = img.shape
            mean = 0
            var = 1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            gauss_min = gauss.min()
            gauss = gauss - gauss_min
            gauss = np.array(255*(gauss / gauss.max()), dtype=np.uint8)
            noisy = (img + gauss) // 2
            cv2.imwrite(dstIMG_path + "Gauss_noise_" + f, noisy)

            # Shifting Left
            shiftedLeft = np.copy(img)
            shift = 0.15
            shiftedLeft[:, 0: img.shape[1] - int(img.shape[1] * shift), :] = img[:,
                                                                             int(img.shape[1] * shift): img.shape[1], :]
            shiftedLeft[:, img.shape[1] - int(img.shape[1] * shift): img.shape[1], :] = 0
            cv2.imwrite(dstIMG_path + "shiftedLeft_" + f, shiftedLeft)

            # Shifting Right
            shiftedRight = np.copy(img)
            shiftedRight[:, int(img.shape[1] * shift): img.shape[1], :] = img[:, 0: img.shape[1] - int(img.shape[1] * shift), :]
            shiftedRight[:, 0: int(img.shape[1] * shift), :] = 0
            cv2.imwrite(dstIMG_path + "shiftedRight_" + f, shiftedRight)

            # Shifting Up
            shiftedUp = np.copy(img)
            shiftedUp[0: img.shape[0] - int(img.shape[0] * shift), :, :] = img[int(img.shape[0] * shift): img.shape[0], :, :]
            shiftedUp[img.shape[0] - int(img.shape[0] * shift): img.shape[0], :, :] = 0
            cv2.imwrite(dstIMG_path + "shiftedUp_" + f, shiftedUp)

            # Shifting Down
            shiftedDown = np.copy(img)
            shiftedDown[int(img.shape[0] * shift): img.shape[0], :, :] = img[ 0: img.shape[0] - int(img.shape[0] * shift), :, :]
            shiftedDown[0: int(img.shape[0] * shift), :, :] = 0
            cv2.imwrite(dstIMG_path + "shiftedDown_" + f, shiftedDown)

            # Sharpness Adjust
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(img, -1, kernel)
            cv2.imwrite(dstIMG_path + "sharpness_" + f, sharp)

            # Brightness Adjust
            brightness = increase_brightness(img, value=30)
            cv2.imwrite(dstIMG_path + "brightness_30_" + f, brightness)

            # Brightness Adjust
            brightness = increase_brightness(img, value=-30)
            cv2.imwrite(dstIMG_path + "brightness_-30_" + f, brightness)

            # Brightness Adjust
            brightness = increase_brightness(img, value=60)
            cv2.imwrite(dstIMG_path + "brightness_60_" + f, brightness)

            # Brightness Adjust
            brightness = increase_brightness(img, value=-60)
            cv2.imwrite(dstIMG_path + "brightness_-60_" + f, brightness)

            # Median Blur
            blurSize = (img.shape[0] + img.shape[1]) // 2 // 20
            if blurSize % 2 == 0:
                blurSize -= 1
            median = cv2.medianBlur(img, blurSize)
            cv2.imwrite(dstIMG_path + "medianBlur%d_" % blurSize + f, median)

            # Gaussian Blur
            Gaussianblur = cv2.GaussianBlur(img, (blurSize, blurSize), 0)
            cv2.imwrite(dstIMG_path + "blur_%dx%d_" % (blurSize, blurSize) + f, Gaussianblur)

            # Low Resolution
            lowResolution = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))
            lowResolution = cv2.resize(lowResolution, (img.shape[1], img.shape[0]))
            cv2.imwrite(dstIMG_path + "lowResolution_" + f, lowResolution)

    def train(self):
        label_counter = 0
        label_list = []
        feature_list = []
        print("total class:", self.n_class)

        for data_dir in self.data_path:
            images = self.read_data_to_4d(data_dir, self.image_num)
            for i in range(len(images)):
                label_list.append(label_counter)
            label_counter += 1

            for img in images:
                # feature extraction and normalization
                feature = self.normalize(
                    self.sess.run(self.h_fc7, feed_dict={self.xs: img.reshape([-1, 224, 224, 3])}).reshape([-1]))
                feature_list.append(feature)

        feature_list = np.array(feature_list)
        label_list = np.array(label_list)
        # shuffling
        shuffled_xs, shuffled_ys = self.shuffle(feature_list, label_list)
        print("shuffling done!")

        clf = svm.SVC(C=10)
        clf.fit(shuffled_xs, shuffled_ys)
        print("training complete!")
        joblib.dump(clf, 'C:/Users/Jerry/PycharmProjects/face/' + 'vgg_fc7_svm.pkl')
        self.clf = clf

    def test(self):
        fileNames = []
        finename = [self.test_folder + f + "/" for f in os.listdir(self.test_folder)]
        labels = []
        for folder in finename:
            _filename = [folder + f for f in os.listdir(folder) if f[-4:] == ".jpg" or f[-4:] == ".png"]
            if int(folder[folder[:-1].rfind("/") + 1: -1][4:]) - 1 < len(self.classfication.keys()) and int(
                    folder[folder[:-1].rfind("/") + 1: -1][4:]) - 1 > -1:
                myKey = int(folder[folder[:-1].rfind("/") + 1: -1][4:]) - 1
            else:
                myKey = len(self.classfication.keys()) - 1

            for f in _filename:
                labels.append(myKey)
                fileNames.append(f)

        fileNames = np.array(fileNames)
        labels = np.array(labels)

        feature_list = []
        for f in fileNames:
            img = cv2.resize(cv2.imread(f, 1), (224, 224))
            img = np.reshape(img, (-1, 224, 224, 3))
            feature = self.sess.run(self.h_fc7, feed_dict={self.xs: img})
            feature_list.append(self.normalize(feature[0]))

        feature_list = np.array(feature_list)

        accuracy = 0.0
        prediction = self.clf.predict(feature_list)
        predictions = np.zeros([len(prediction), self.n_class], dtype=np.int32)
        for i in range(len(prediction)):
            predictions[i, prediction[i]] = 1
            if prediction[i] == labels[i]:
                accuracy += 1.0
        accuracy /= len(prediction)
        print(accuracy)

    def test_real_time(self, img, pre_load=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            if pre_load:  # pre-load and access memory space
                img = np.array(cv2.resize(img, (224, 224)), dtype=np.float32).reshape(-1, 224, 224, 3)
                feature = self.sess.run(self.h_fc7, feed_dict={self.xs: img})
                print("Model Pre-loaded Completely")
                return

            rect, landmark = self.get_detect(gray)
            if len(rect) is not 0:
                img_list = []
                area = np.zeros(shape=len(rect))
                img_counter = 0
                for (x, y, w, h) in rect:
                    area[img_counter] = w * h
                    img_counter += 1

                    crop_img = img[y:y + h, x:x + w, :]
                    crop_img = cv2.resize(crop_img, (224, 224))
                    img_list.append(crop_img)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

                img_list = np.reshape(np.asarray(img_list), newshape=(-1, 224, 224, 3))

                maxval, idx = self.Maximum(area)

                feature = self.sess.run(self.h_fc7, feed_dict={self.xs: img_list})
                feature = self.normalize(feature)

                img_counter = 0
                for (x, y, w, h) in rect:
                    prediction = self.clf.predict(feature[img_counter].reshape(1, -1))
                    cv2.putText(img, self.classfication[prediction[0]], (x, y + h - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)
                    if img_counter == idx:
                        if prediction[0] != len(self.classfication) - 1:
                            self.counter += 1
                            if self.counter > 3:
                                self.driver_ID = "Driver"
                                self.counter = 0
                        else:
                            self.counter = 0
                            self.driver_ID = "Stanger"
                    img_counter += 1
        except:
            print("no face detected !")
        finally:
            if not pre_load:
                cv2.imshow('FaceRecognition', img)
                cv2.waitKey(1)

    def Maximum(self, input_array):
        """find max in 1D-array"""
        max_val = input_array[0]
        idx = 0
        for i in range(len(input_array)):
            if input_array[i] > max_val:
                max_val = input_array[i]
                idx = i
        return max_val, idx

    def get_detect(self, gray):
        try:
            rects = self.detector(gray, 1)
            rect_list = []
            shape_list = []
            for (i, rect) in enumerate(rects):
                shape = self.predictor(gray, rect)
                shape_list.append(face_utils.shape_to_np(shape))
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                rect_list.append((x, y, w, h))
            return np.array(rect_list), np.array(shape_list)
        except:
            print("detect None")

    def get_landmarks(self, img):
        eye_size = (120, 60)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rect, shape = self.get_detect(gray)
            area = np.zeros(len(rect))
            for i in range(len(rect)):
                area[i] = rect[i][2] * rect[i][3]
                x = rect[i][0]
                y = rect[i][1]
                w = rect[i][2]
                h = rect[i][3]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            maxval, idx = self.Maximum(area)

            xlist = []
            ylist = []
            for (_x, _y) in shape[idx]:
                # cv2.circle(gray, (_x, _y), 1, (0, 0, 255), -1)
                xlist.append(_x)
                ylist.append(_y)
            left_eyeX = np.array(xlist[36:42])
            left_eyeY = np.array(ylist[36:42])
            right_eyeX = np.array(xlist[42:48])
            right_eyeY = np.array(ylist[42:48])
            left_center = (int(np.mean(left_eyeX)), int(np.mean(left_eyeY)))
            right_center = (int(np.mean(right_eyeX)), int(np.mean(right_eyeY)))

            l_EAR = (math.sqrt((left_eyeX[1] - left_eyeX[5])**2 + (left_eyeY[1] - left_eyeY[5])**2) +
                       math.sqrt((left_eyeX[2] - left_eyeX[4]) ** 2 + (left_eyeY[2] - left_eyeY[4]) ** 2)) / \
                      (2 * math.sqrt((left_eyeX[0] - left_eyeX[3])**2 + (left_eyeY[0] - left_eyeY[3])**2))

            r_EAR = (math.sqrt((right_eyeX[1] - right_eyeX[5]) ** 2 + (right_eyeY[1] - right_eyeY[5]) ** 2) +
                     math.sqrt((right_eyeX[2] - right_eyeX[4]) ** 2 + (right_eyeY[2] - right_eyeY[4]) ** 2)) / \
                    (2 * math.sqrt((right_eyeX[0] - right_eyeX[3]) ** 2 + (right_eyeY[0] - right_eyeY[3]) ** 2))
            # print('EAR:', l_EAR, r_EAR)

            if (l_EAR + r_EAR)/2 < 0.2:
                self.close_eye_counter += 1
                if self.close_eye_counter > 3:
                    self.close_eye_flag = True
                    print('sleep')
            else:
                self.close_eye_counter = 0
                self.close_eye_flag = False
        except:
            pass
        finally:
            cv2.imshow('eye detector', img)
            cv2.waitKey(1)


def rotate(image, angle, center=None, scale=1.0):
    # 作者：Eric_AIPO
    # 链接：https: // www.jianshu.com / p / b5c29aeaedc7
    # 來源：简书
    # 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。


    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.array(v, dtype=np.int16)

    if value < 0:
        lim = abs(value)
        v[v < lim] = 0
        v[v >= lim] += value
    else:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

    v = np.array(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
