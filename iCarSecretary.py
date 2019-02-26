import cv2
import os
import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask, request, abort
from io import BytesIO
import time
from car_line_class import Lane
from FaceRecognition import Neural_Network
import threading
import multiprocessing as mp
import serial
import json
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, StickerSendMessage
)

# initialize our server
sio = socketio.Server()
app = Flask(__name__)
Lineapp = Flask(__name__)

line_bot_api = LineBotApi('0EzDM8picIHDKj2jByE3PzkCh2Y0EWmYQhHZm+8XmD2mtxCMH12sNmnMJKn5xJfjBoxrb87WTVCN0rQxiiRodTjn4YfFF9r68RcmY8WD1tuBN1V5d89pO+5kihOKk6O3cU1MEh2rXlPB//rFMgHiWAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('b51194a933addffd4212f0af6607a0fa')

n_input = 192
n_step = 108
state = False
start_flag = True
close_eye_flag = False
danger_flag = False
ldw_flag = False
danger_counter = 0
driver_id = False
close_flag = True
camUsed = 0
previous_time = time.time()
model_state = 0         # 0: restore model, 1: access memory, 2: model loaded completed

timeout = 15
serialPort = 'COM3'
serialBaud = 115200
serialTimeOut = 0.5
code = ""
correct_code = False
training = False

user_id_list = []                         # user line ID
USERID = {"user_id_list": user_id_list}
FileOpened = False
num_user = len(user_id_list)


def init_user_id():
    global USERID, user_id_list, num_user, FileOpened
    if not FileOpened:
        FileOpened = True
        if not os.path.exists("./USERID.json"):
            with open('./USERID.json', 'w') as fp:
                json.dump(USERID, fp)
        USERID = json.loads(open("./USERID.json").read())
        try:
            FileOpened = False
            user_id_list = USERID["user_id_list"]
            num_user = len(user_id_list)
            print(user_id_list)
        except FileNotFoundError:
            FileOpened = False
            raise FileNotFoundError("No such file exists!!!")


@Lineapp.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    Lineapp.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global USERID, user_id_list, num_user, FileOpened, code, correct_code
    try:
        user_id = event.source.user_id
        profile = line_bot_api.get_profile(event.source.user_id)

        if "註冊" not in event.message.text:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=event.message.text))
        else:
            if "取消註冊" in event.message.text:

                if user_id not in user_id_list:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您未曾註冊過，請問註冊ㄇ？"))

                else:
                    user_id_list.remove(event.source.user_id)
                    num_user = len(user_id_list)
                    USERID = {"user_id_list": user_id_list}
                    FileOpened = True
                    with open('./USERID.json', 'w') as fp:
                        json.dump(USERID, fp)
                    FileOpened = False
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您已取消註冊"))

            elif "註冊" in event.message.text:
                if user_id not in user_id_list:
                    user_id_list.append(event.source.user_id)
                    num_user = len(user_id_list)
                    USERID = {"user_id_list": user_id_list}
                    FileOpened = True
                    with open('./USERID.json', 'w') as fp:
                        json.dump(USERID, fp)
                    FileOpened = False
                    code = ""
                    correct_code = False
                    while "000" == code or code == "":
                        code = ""
                        for i in range(3):
                            code += str(np.random.randint(0, 2))
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請於車載觸控面板輸入密碼: " + code))

                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您已註冊過"))

        print("**%s(%s) said %s" % (profile.display_name, user_id, event.message.text))
    except:
        print("**Not a friend of the Bot")


def Update(l, d):
    """
    Update Global Variables within two process
    :param l: multiProcess locker
    :param d: multiProcess shared variables dictionary
    """
    global state, start_flag, close_eye_flag, danger_flag, driver_id, previous_time, close_flag, model_state, close_flag, correct_code, training
    while True:
        ### if Update() receive data ###
        if d['dataRX']:
            state = d['state']
            start_flag = d['start_flag']
            driver_id = d['driver_id']
            close_eye_flag = d['close_eye_flag']
            danger_flag = d['danger_flag']
            model_state = d['model_state']
            correct_code = d['correct_code']
            training = d['training']
            d['dataRX'] = False
        else:
            if d['state'] != state or d['start_flag'] != start_flag or d['driver_id'] != driver_id or d['previous_time'] != previous_time or d['close_eye_flag'] != close_eye_flag or d['danger_flag'] != danger_flag or d['close_flag'] != close_flag or d['correct_code'] != correct_code:
                l.acquire()
                d['state'] = state
                d['previous_time'] = previous_time
                # d['driver_id'] = driver_id
                # d['close_eye_flag'] = close_eye_flag
                d['correct_code'] = correct_code
                d['close_flag'] = close_flag
                d['start_flag'] = start_flag
                d['danger_flag'] = danger_flag
                d['dataTX'] = True
                l.release()


def lane_detect(img):
    """
    Road lane detection
    :param img: road image
    :return: None
    """
    global danger_flag, danger_counter, ldw_flag

    img = cv2.resize(img, (int(n_input), int(n_step)), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only white colors
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 20])

    # my yellow
    # Threshold the HSV image to get only yellow colors
    lower_yellow = np.array([25, 150, 60])
    upper_yellow = np.array([35, 255, 225])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv2.GaussianBlur(mask_yellow, (3, 3), 0)
    mask_black = cv2.medianBlur(mask_black, 3)
    cv2.imshow('mask_yellow', mask_yellow)
    cv2.imshow('mask_black', mask_black)
    mask = cv2.bitwise_or(mask_black, mask_yellow)
    LoG_kernel = np.array([[0, -4, 0], [-4, 32, -4], [0, -4, 0]])
    edges = cv2.filter2D(mask, -1, LoG_kernel)
    edges = cv2.medianBlur(edges, 7)
    # edges = cv2.Canny(edges, 50, 150)
    cv2.imshow('edges', edges)

    vertices = np.array([[(int(edges.shape[1] * 0), int(edges.shape[0] * 1)),
                          (int(edges.shape[1] * 0), int(edges.shape[0] * 0.45)),
                          (int(edges.shape[1] * 1), int(edges.shape[0] * 0.45)),
                          (int(edges.shape[1] * 1), int(edges.shape[0] * 1))]],
                        dtype=np.int32)
    lane = Lane(edges)
    imCrop = lane.region_of_interest(vertices)
    cv2.imshow('ROI_crop', imCrop)

    # hough transform
    imLine = lane.hough_lines(rho=1, theta=np.pi / 180, threshold=25, min_line_len=20, max_line_gap=10,
                              midLoc=(vertices[0][1][0] + vertices[0][2][0]) // 2, order=1)
    cv2.imshow('hough', imLine)
    imProc = cv2.addWeighted(img, 0.8, imLine, 1, 0)
    cv2.imshow('frame', imProc)
    if lane.danger:
        print("dangerous")
        danger_flag = True
        ldw_flag = True
        danger_counter = 0
    else:
        if danger_counter < 3:
            danger_counter += 1
        else:
            danger_counter = 0
            danger_flag = False
            ldw_flag = False
    return


@sio.on('telemetry')        # read data
def telemetry(sid, data):
    """
    Receive WebSocket data from Unity simulator
    :param sid: Socket ID
    :param data: Socket Data
    :return: None
    """
    global state
    global start_flag
    global driver_id
    global danger_flag
    global previous_time
    global close_eye_flag
    global danger_counter
    global close_flag
    global ldw_flag

    if data:
        if int(data["close"]) == 1:
            state = False
            start_flag = True
            print("closed ", sid)
            driver_id = False
            danger_flag = False
            ldw_flag = False
            close_eye_flag = False
            close_flag = True
            danger_counter = 0
            sio.emit('close', data={}, skip_sid=True)
            return
        try:
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image = np.asarray(image)  # from PIL image to numpy array
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('SrcImg', img)
            lane_detect(img)
            cv2.waitKey(1)

        except Exception as e:
            print(e)
        # print("time interval:", time.time() - previous_time)
        previous_time = time.time()
        if danger_flag or close_eye_flag:
            send_control(driver_id=driver_id, danger="1")
        else:
            send_control(driver_id=driver_id, danger="0")


@sio.on('connect')
def connect(sid, environ):
    """
    When Unity simulator connected to server via WebSocket
    :param sid: Socket ID
    """
    global state
    global start_flag
    global danger_flag
    global driver_id
    global previous_time
    global danger_counter
    global ldw_flag
    global close_flag

    print("connect ", sid)
    state = True
    start_flag = True
    danger_flag = False
    driver_id = False
    ldw_flag = False
    close_flag = False
    danger_counter = 0
    previous_time = time.time()
    send_control(driver_id=driver_id)


def send_control(driver_id=False, danger="0"):
    """
    Send Control to Unity simulator
    :param driver_id: Driver ID from Face Recognition
    :param danger: Dangerous alarm from close eye detection and land detection
    """
    if driver_id:
        sio.emit(
            "steer",
            data={
                'Driver_ID': "Driver"
                , 'Danger': danger
            },
            skip_sid=True)
    else:
        sio.emit(
            "steer",
            data={
                'Driver_ID': "Stranger"
                , 'Danger': danger
            },
            skip_sid=True)


def face(l, d, __state, __start_flag, __driver_id, __previous_time, __close_eye_flag, __danger_flag, __model_state, __close_flag, __correct_code, __training):
    """
    Face Recognition with VGG-16 pre-trained model
    Prevent car stolen
    :param l: multiProcess locker
    :param d: multiProcess shared variables dictionary
    :param __state: Face Recognition and Close Eye Detection states in mp dictionary
    :param __start_flag: Connection state in mp dictionary
    :param __driver_id: Driver ID from Face Recognition
    :param __previous_time: Time clock for timeout used
    :param __close_eye_flag: Close eye detection flag
    :param __danger_flag: Close eye or LDW dangerous flag
    :param __model_state: model pre-load state
    :param __close_flag: Unity simulator closed
    """
    __model_state.value = 0
    d['model_state'] = 0
    d['dataRX'] = True

    nn = Neural_Network()

    __model_state.value = 1
    d['model_state'] = 1
    d['dataRX'] = True

    cap = cv2.VideoCapture(nn.CAM_FLAG)
    ret, img = cap.read()
    nn.test_real_time(img, pre_load=True)   # pre-load and access memory space

    __model_state.value = 2
    d['model_state'] = 2
    d['dataRX'] = True

    while True:
        if d['dataTX']:
            __state.value = d['state']
            __start_flag.value = d['start_flag']
            __driver_id.value = d['driver_id']
            __previous_time.value = d['previous_time']
            __close_eye_flag.value = d['close_eye_flag']
            __danger_flag.value = d['danger_flag']
            __close_flag.value = d['close_flag']
            __correct_code.value = d['correct_code']
            d['dataTX'] = False

        if __correct_code.value:
            __training.value = True
            l.acquire()
            d['training'] = True
            d['dataRX'] = True
            l.release()

            print('training...')
            img_number = nn.image_num // 10
            _img_counter = 1
            while _img_counter < img_number:
                ret, img = cap.read()
                if ret:
                    n_face = nn.data_collection(img, _img_counter)
                    if n_face is not 0:
                        _img_counter += 1

            nn.data_augmentation()      # data augmentation 50->500 images

            nn.train()                  # vgg16->svm training
            print('training complete')
            l.acquire()
            __correct_code.value = False
            __training.value = False
            d['correct_code'] = False
            d['training'] = False
            d['dataRX'] = True
            l.release()
        if __state.value and __start_flag.value:
            nn.driver_ID = "Stranger"
            nn.counter = 0
            while __state.value and __start_flag.value:
                if d['dataTX']:
                    __state.value = d['state']
                    __start_flag.value = d['start_flag']
                    __driver_id.value = d['driver_id']
                    __previous_time.value = d['previous_time']
                    __close_eye_flag.value = d['close_eye_flag']
                    __danger_flag.value = d['danger_flag']
                    __close_flag.value = d['close_flag']
                    __correct_code.value = d['correct_code']
                    d['dataTX'] = False
                if not __state.value or not __start_flag.value:
                    break
                ret, img = cap.read()
                if ret:
                    nn.test_real_time(img)
                    if nn.driver_ID == "Driver":
                        l.acquire()
                        __driver_id.value = True
                        d['driver_id'] = True
                        __start_flag.value = False
                        d['start_flag'] = False
                        d['dataRX'] = True
                        l.release()
                        print(nn.driver_ID)

                    if (time.time() - __previous_time.value) > timeout:
                        print("time out")
                        __state.value = False
                        __driver_id.value = False
                        __close_eye_flag.value = False
                        l.acquire()
                        d['driver_id'] = False
                        d['state'] = False
                        d['close_eye_flag'] = False
                        d['dataRX'] = True
                        l.release()

        if not __start_flag.value and __state.value:
            while __state.value:
                if d['dataTX']:
                    __state.value = d['state']
                    __start_flag.value = d['start_flag']
                    __driver_id.value = d['driver_id']
                    __previous_time.value = d['previous_time']
                    __close_eye_flag.value = d['close_eye_flag']
                    __danger_flag.value = d['danger_flag']
                    __close_flag.value = d['close_flag']
                    d['dataTX'] = False
                if not __state.value:
                    break
                ret, img = cap.read()
                if ret:
                    nn.get_landmarks(img)
                    l.acquire()
                    __close_eye_flag.value = nn.close_eye_flag
                    d['close_eye_flag'] = __close_eye_flag.value
                    __danger_flag.value = __close_eye_flag.value
                    d['danger_flag'] = __danger_flag.value

                    if (time.time() - __previous_time.value) > timeout:
                        print("time out")
                        __state.value = False
                        __driver_id.value = False
                        __close_eye_flag.value = False
                        d['driver_id'] = False
                        d['state'] = False
                        d['close_eye_flag'] = False
                    d['dataRX'] = True
                    l.release()
            cv2.destroyAllWindows()
        nn.close_eye_flag = False
        nn.close_eye_counter = 0
        nn.driver_ID = "Stranger"
        nn.counter = 0
        __driver_id.value = False
        __close_eye_flag.value = False
        d['driver_id'] = False
        d['close_eye_flag'] = False
        d['dataRX'] = True


def socket_server():
    """
    create WebSocket Server
    """
    global app
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


def serial_server(__driver_id, __close_eye_flag, __model_state):
    """
    Communicate with Embedded GUI system via UART
    :param driver_id: Driver ID from Face Recognition
    :param close_eye_flag: Close Eye Detection flag
    :param ldw_flag: LDW dangerous flag
    :param model_state: model pre-load state
    """
    global ldw_flag, close_flag, user_id_list, code, correct_code, training

    while True:
        try:
            sp = serial.Serial(port=serialPort, baudrate=serialBaud, timeout=serialTimeOut, bytesize=serial.EIGHTBITS,
                               stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE)
            serial_msg = sp.readline()
            sp.write(b'00000000')
            break
        except:
            pass

    driver_id_counter = False
    while True:
        try:
            sp_onehot_value = 0
            if __model_state.value == 2:      # after pre-loaded, then detection face, lane and eyes
                if not close_flag:
                    if __driver_id.value:
                        sp_onehot_value += pow(2, 7)
                        if not driver_id_counter:
                            line_bot_api.multicast(user_id_list, TextSendMessage(text="Welcome back Driver"))
                            driver_id_counter = True
                    if ldw_flag:
                        sp_onehot_value += pow(2, 6)
                        line_bot_api.multicast(user_id_list, TextSendMessage(text="Please Drive between the lines"))
                    if __close_eye_flag.value:
                        sp_onehot_value += pow(2, 5)
                        line_bot_api.multicast(user_id_list, TextSendMessage(text="Don't take a nap"))
                else:
                    __driver_id.value = False
                    ldw_flag = False
                    __close_eye_flag.value = False
                    driver_id_counter = False

                sp_onehot_value += pow(2, 4)

                if len(code) is 3:
                    for i in range(3):
                        if int(code[i]) == 1:
                            sp_onehot_value += pow(2, 3-i)

                if training:
                    sp_onehot_value += pow(2, 0)

            tmp = bin(sp_onehot_value)[2:]
            length = len(bin(sp_onehot_value)) - 2
            if length <= 8:
                for i in range(8-length):
                    tmp = "0" + tmp
            tmp = tmp.encode()
            sp.write(tmp)
            serial_msg = sp.readline()
            if len(serial_msg) is 8:
                if int(serial_msg[-1]) == 49:
                    correct_code = True
                else:
                    correct_code = False
            else:
                correct_code = False
        except:
            while True:
                try:
                    sp = serial.Serial(port=serialPort, baudrate=serialBaud, timeout=serialTimeOut,
                                       bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE,
                                       parity=serial.PARITY_NONE)
                    serial_msg = sp.readline()
                    sp.write(b'00000000')
                    break
                except:
                    pass


def line_server():
    Lineapp.run(debug=False)


if __name__ == '__main__':
    init_user_id()
    lock = mp.Lock()
    manager = mp.Manager()
    Dict = manager.dict()
    Dict['dataRX'] = False
    Dict['dataTX'] = False
    Dict['state'] = state
    Dict['start_flag'] = start_flag
    Dict['close_eye_flag'] = close_eye_flag
    Dict['danger_flag'] = danger_flag
    Dict['driver_id'] = driver_id
    Dict['previous_time'] = previous_time
    Dict['model_state'] = model_state
    Dict['close_flag'] = close_flag
    Dict['correct_code'] = correct_code
    Dict['training'] = training
    _state = mp.Value('I', state)
    _start_flag = mp.Value('I', start_flag)
    _close_eye_flag = mp.Value('I', close_eye_flag)
    _danger_flag = mp.Value('I', danger_flag)
    _driver_id = mp.Value('I', driver_id)
    _previous_time = mp.Value('d', previous_time)
    _model_state = mp.Value('I', model_state)
    _close_flag = mp.Value('I', close_flag)
    _correct_code = mp.Value('I', correct_code)
    _training = mp.Value('I', training)
    p1 = mp.Process(target=face, name="P1", args=(lock, Dict, _state, _start_flag, _driver_id, _previous_time, _close_eye_flag, _danger_flag, _model_state, _close_flag, _correct_code, _training))
    t2 = threading.Thread(target=socket_server, name="T2")
    t3 = threading.Thread(target=Update, name='T3', args=(lock, Dict))
    t4 = threading.Thread(target=serial_server, name='T4', args=(_driver_id, _close_eye_flag, _model_state))
    t5 = threading.Thread(target=line_server, name='T5')
    p1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
