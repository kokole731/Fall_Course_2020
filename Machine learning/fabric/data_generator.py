import os
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import const as C

label_dir = os.listdir(os.path.join(C.PATH_FABRIC, 'label_json'))

flaw_types = (1, 2, 14)

# if not os.path.exists('data_processed2'):
#     os.mkdir('data_processed2')
# for flaw_type in flaw_types:
#     os.makedirs(os.path.join('data_processed2', str(flaw_type)))



for label in label_dir:
    cwd = os.path.join(C.PATH_FABRIC, 'label_json', label)
    for json_file in os.listdir(cwd):
        with open(os.path.join(cwd, json_file)) as f:
            data = json.load(f)
            flaw_type = data['flaw_type']
            if not flaw_type in flaw_types:
                continue
            temp_img = os.path.join(C.PATH_FABRIC, 'temp', label, json_file.replace('json', 'jpg'))
            trgt_img = os.path.join(C.PATH_FABRIC, 'trgt', label, json_file.replace('json', 'jpg'))
            temp_data = cv2.imread(temp_img)
            trgt_data = cv2.imread(trgt_img)
            if temp_data.shape != trgt_data.shape:
                print('error')
                continue
            err_data = temp_data - trgt_data
            cv2.imwrite(os.path.join(C.PATH_IMAGE, str(flaw_type), json_file.replace('json', 'jpg')), err_data)
            # plt.imshow(err_data)
            # plt.show()
            # data['img'] = err_data
            # np.save(os.path.join('data_processed', str(flaw_type), json_file.split('.')[0]), data)


# img_good = cv2.imread('good.jpg')
# img_fault = cv2.imread('fault.jpg')
# img_err = img_good - img_fault

# cv2.imshow('img_fault', img_fault)
# cv2.imshow('img_good', img_good)

# plt.imshow(img_err)
# plt.show()