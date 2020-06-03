import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, LSTM
from keras.layers import MaxPooling2D
from keras.layers import TimeDistributed


DICT = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Red", "Women", "Enemy", "Son",
        "Man", "Away", "Drawer", "Born", "Learn", "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food",
        "Argentina", "Uruguay", "Country", "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo", "Hungry",
        "Map", "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue",
        "Candy", "Chewing-gum", "Spaghetti", "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch",
        "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"
        ]


def prepare_video_capture(path):
    frame_count = 10
    frame_num = 0

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    frame_step = int(length / frame_count)  # every frame_step's frame is taken

    cur_sign = []

    while cap.isOpened():
        frame_num += 1
        if not cap.grab():
            break
        if frame_num % frame_step != 0:
            continue
        ret, frame = cap.retrieve()
        masked_grey = parse_frame(frame, (60, 160))

        cur_sign.append(masked_grey)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(cur_sign[:frame_count])


def get_small_model():
    use_dropout = True
    num_classes = 10

    small_model = Sequential()
    small_model.add(TimeDistributed(Conv2D(filters=32,
                                           kernel_size=(5, 5),
                                           activation='relu',
                                           input_shape=(10, 212, 380, 1))))
    small_model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    if use_dropout:
        small_model.add(TimeDistributed(Dropout(0.25)))
    small_model.add(TimeDistributed(Conv2D(filters=32,
                                           kernel_size=(3, 3),
                                           activation='relu')))
    small_model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    if use_dropout:
        small_model.add(TimeDistributed(Dropout(0.25)))

    small_model.add(TimeDistributed(Flatten()))

    small_model.add(LSTM(64, dropout=0.25))

    if use_dropout:
        small_model.add(Dropout(0.25))

    small_model.add(Dense(256, activation='relu'))
    small_model.add(Dense(num_classes, activation='softmax'))

    small_model.build(input_shape=(None, 10, 212, 380, 1))
    small_model = load_model("small_model_weights.h5")
    return small_model


def get_model():
    input_shape = (10, 212, 380, 1)  # need 10 frames (212x380)
    use_dropout = True
    num_classes = 16

    model_for_16 = Sequential()
    model_for_16.add(TimeDistributed(Conv2D(filters=32,
                                            kernel_size=(5, 5),
                                            activation='relu',
                                            input_shape=input_shape)))
    model_for_16.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    if use_dropout:
        model_for_16.add(TimeDistributed(Dropout(0.25)))
    model_for_16.add(TimeDistributed(Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            activation='relu')))
    model_for_16.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    if use_dropout:
        model_for_16.add(TimeDistributed(Dropout(0.25)))

    model_for_16.add(TimeDistributed(Flatten()))

    model_for_16.add(LSTM(64))

    if use_dropout:
        model_for_16.add(Dropout(0.25))

    model_for_16.add(Dense(256, activation='relu'))
    model_for_16.add(Dense(num_classes, activation='softmax'))

    model_for_16.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )

    model_for_16.load_weights('model_16.h5')
    return model_for_16


model = get_small_model()


def predict(model, sign):
    classes = model.predict_classes(sign)
    return classes[0]


def apply_mask(frame, color, gap=10, lower_s=50, lower_v=50, upper_s=255, upper_v=255):
    """
    :param: color - must be an integer representing H-value of color in HSV format
    :return: resulting mask
    """
    # add some gap
    lower_bound = np.array([color - gap, lower_s, lower_v])
    upper_bound = np.array([color + gap, upper_s, upper_v])

    # first num є [0, 180]
    if (lower_bound[0] < 0):
        lower_bound[0] = 0
    if (upper_bound[0] > 180):
        upper_bound[0] = 180
    return cv2.inRange(frame, lower_bound, upper_bound)


def parse_frame(frame, colors):
    """
    :param frame: original frame
    :param colors: iterable of integers represinting H value of the color
    :return: parsed frame
    """
    frame_size = (380, 212)
    resized = cv2.resize(frame, frame_size)  # resize frame

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)  # from RGB to HSV

    masks = []
    for h in colors:
        masks.append(apply_mask(hsv, h))

    mask = sum(masks)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Segmenting the cloth out of the frame using bitwise_and with the inverted mask
    masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    h, s, masked_grey = cv2.split(masked)
    return masked_grey


def real_time(h1, h2):
    FRAME_RATE = 10

    camera = cv2.VideoCapture(0)
    cur_sign = np.zeros((10, 212, 380))
    frame_num = 0
    sign = 0

    while camera.isOpened():
        ret, frame = camera.read()
        # frame = cv2.flip(frame, 1)  # flip the frame horizontally
        colors = []

        if h1 is not None:
            colors.append(h1)

        if h2 is not None:
            colors.append(h2)

        masked_grey = parse_frame(frame, colors)

        # ========================== process frame ========================== #
        frame_num += 1
        if frame_num % FRAME_RATE == 0:
            frame_num = 0
            cur_sign[:9] = cur_sign[1:]
            cur_sign[9] = masked_grey

            pred_sign = np.expand_dims(cur_sign.reshape(*cur_sign.shape, 1), axis=0)

            sign = predict(model, pred_sign)

            cv2.putText(masked_grey, f"Prediction: {DICT[sign]}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))

        cv2.putText(masked_grey, f"Prediction: {DICT[sign]}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.imshow('masked', masked_grey)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
