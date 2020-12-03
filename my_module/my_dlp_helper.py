import os
import xmlrpc.client
import json
import shutil
import heapq
import cv2
import sys
import pickle
from my_module.my_preprocess import my_preprocess
import my_config
from my_module import db_helper


#region get Server CAM and deeplift URL, Now ajax, saliency.py

def get_server_cam_big_class_multi_class_url():
    url = "http://localhost:23000/"
    return url

def get_server_cam_big_class_multi_label_url():
    url = "http://localhost:23001/"
    return url

#do not generate subclass heatmap now
'''
def get_server_cam_sub_class_url(bigclass, subclass):
    if bigclass == 0 and subclass == 2:  #DR1
        url = "http://localhost:23100/"
    if bigclass == '1':  #DR
        url = "http://localhost:23000/"
    if bigclass == '1':  #RVO
        url = "http://localhost:23000/"
    if bigclass == '29': # BLUR
        url = "http://localhost:23000/"

    return url
'''

def get_deeplift_big_class_url(model_no):
    if model_no == 0:
        url = "http://localhost:25000/"
    if model_no == 1:
        url = "http://localhost:25001/"

    return url

#do not generate subclass heatmap now
'''
def get_deeplift_sub_class_url(bigclass):
    if bigclass == '0.2':  # DR1
        url = "http://localhost:23000/"
    if bigclass == '1':  # DR
        url = "http://localhost:23000/"
    if bigclass == '1':  # RVO
        url = "http://localhost:23000/"
    if bigclass == '29':  # BLUR
        url = "http://localhost:23000/"

    return url
'''

#endregion

#根据疾病编号获取疾病名称
def get_disease_name(no, class_type='bigclass', lang='en'):
    #  sys.path[0]  #MyWeb dir
    json_dir = os.path.join(sys.path[0], 'diseases_json')

    if class_type == 'm2':
        json_file = 'classid_to_human_m2.json'
    elif class_type == 'm1':
        json_file = 'classid_to_human_m1.json'
    elif class_type == 'left_right':
        json_file = 'classid_to_human_left_right.json'
    elif class_type == 'gradable':
        json_file = 'classid_to_human_gradable.json'
    elif class_type == 'bigclass':
        json_file = 'classid_to_human_bigclass.json'
    elif class_type == 'subclass_0_1':
        json_file = 'classid_to_human_subclass0_1.json'
    elif class_type == 'subclass_0_2':
        json_file = 'classid_to_human_subclass0_2.json'
    elif class_type == 'subclass_0_3':
        json_file = 'classid_to_human_subclass0_3.json'
    elif class_type == 'subclass_1':
        json_file = 'classid_to_human_subclass1.json'
    elif class_type == 'subclass_2':
        json_file = 'classid_to_human_subclass2.json'
    elif class_type == 'subclass_5':
        json_file = 'classid_to_human_subclass5.json'
    elif class_type == 'subclass_10':
        json_file = 'classid_to_human_subclass10.json'
    elif class_type == 'subclass_15':
        json_file = 'classid_to_human_subclass15.json'
    elif class_type == 'subclass_29':
        json_file = 'classid_to_human_subclass29.json'

    if lang == 'cn':       #全局变量
        json_file = 'cn_' + json_file


    json_file = os.path.join(json_dir, json_file)

    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
        for i in range(len(data['diseases'])):
            if data['diseases'][i]['NO'] == no:
                return data['diseases'][i]['NAME']


# set RPC Service Port, set parameters(multi-class,multi-label) and get results
def predict_single_class(img_source, class_type='0', softmax_or_sigmoids='softmax'):

    if class_type == '-5':  # img_position, macular center, optic_disc center, other
        server_port = my_config.PORT_BASE - 5     #19995
    elif class_type == '-4':  # left_right
        server_port = my_config.PORT_BASE - 4     #19996
    elif class_type == '-3':  # gradable
        server_port = my_config.PORT_BASE - 3     #19997
    elif class_type == '-2':
        server_port = my_config.PORT_BASE - 2    #19998 dirty_lens
    elif class_type == '-1':
        server_port = my_config.PORT_BASE - 1   #19999 ocular_surface

    elif class_type == '0':
        server_port = my_config.PORT_BASE     #20000 BigClass multi-class
    elif class_type == '0_10':
        server_port = my_config.PORT_BASE_MULTI_LABEL     #20000 BigClass multi-label
    elif class_type == '0_1':
        server_port = my_config.PORT_BASE + 1  #20001
    elif class_type == '0_2':
        server_port = my_config.PORT_BASE + 2   #20002
    elif class_type == '0_3':
        server_port = my_config.PORT_BASE + 3   #20003
    else:
        server_port = my_config.PORT_BASE + int(class_type) * 10

    SERVER_URL = 'http://localhost:{0}/'.format(server_port)

    with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
        if softmax_or_sigmoids == 'softmax':
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_softmax(
                img_source)
        else: # Multi-label is different from Multi-class:  pred correct_model_no are lists
            # predict_sigmoids(img1, preproess=False, neglect_class0=False)
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_sigmoids(
                img_source, False, False)

        return prob_list, pred_list, prob_total, pred_total, correct_model_no


#调用检测视盘服务 mask=True MaskRCNN, mask=False Retinant
def detect_optic_disc(img_source, mask=False):
    server_port = my_config.PORT_BASE + 1000  #21000

    SERVER_URL = "http://localhost:" + str(server_port) + "/"

    with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
        if not mask:
            found_optic_disc, img_file_crop = \
                proxy1.detect_optic_disc(img_source)

            return found_optic_disc, img_file_crop
        else:
            found_optic_disc, img_file_crop, img_file_crop_mask = \
                proxy1.detect_optic_disc_mask(img_source)

            return found_optic_disc, img_file_crop, img_file_crop_mask


# save files different from predict_all_multi_class, so seperate
'''
resized_384.jpg
preprocessed_384.jpg
img_file_preprocessed_448.jpg
preprocessed_512.jpg

optic_disc_112.jpg
optic_disc_mask_112.jpg

predict_result.pkl

'''

def predict_all_pacs(checkno, file_img_source, baseDir, lang,
                     cam_type='1', show_deeplift=False, show_deepshap=False):
    result_s = ''
    predict_result = {}  # save to PKL

    # one checkno  multiple files, create one sub directory for one file
    _, filename = os.path.split(file_img_source)
    filename_base, filename_ext = os.path.splitext(filename)
    baseDir = os.path.join(baseDir, checkno, filename_base)
    os.makedirs(os.path.dirname(baseDir), exist_ok=True)

    predict_result['str_uuid'] = checkno + '_', filename_base

    predict_result['cam_type'] = cam_type
    predict_result['show_deeplift'] = show_deeplift
    predict_result['show_deepshap'] = show_deepshap

    # region image preprocess, image resize
    img_source = cv2.imread(file_img_source)

    # 原始图像resize 384*384
    IMAGESIZE = 384
    img_file_resized_384 = os.path.join(baseDir, 'resized_384.jpg')
    img_resized = cv2.resize(img_source, (IMAGESIZE, IMAGESIZE))
    cv2.imwrite(img_file_resized_384, img_resized)
    predict_result['img_file_resized_384'] = img_file_resized_384

    # subclass 10, optic disc
    img_file_preprocessed_512 = os.path.join(baseDir, 'preprocessed_512.jpg')
    my_preprocess(img_source, crop_size=512,
                  img_file_dest=img_file_preprocessed_512, add_black_pixel_ratio=0)
    img_preprocess_512 = cv2.imread(img_file_preprocessed_512)
    cv2.imwrite(img_file_preprocessed_512, img_preprocess_512)

    # DR1
    img_file_preprocessed_448 = os.path.join(baseDir, 'img_file_preprocessed_448.jpg')
    img_preprocess_448 = cv2.resize(img_preprocess_512, (448, 448))
    cv2.imwrite(img_file_preprocessed_448, img_preprocess_448)
    predict_result['img_file_preprocessed_448'] = img_file_preprocessed_448

    # others use 384*384
    img_file_preprocessed_384 = os.path.join(baseDir, 'preprocessed_384.jpg')
    img_preprocess_384 = cv2.resize(img_preprocess_512, (384, 384))
    cv2.imwrite(img_file_preprocessed_384, img_preprocess_384)
    predict_result['img_file_preprocessed_384'] = img_file_preprocessed_384

    # endregion

    disease_name = ''

    # region gradable(-3), left_right(-4), position(-5), dirty_lens(-2), ocular_surface(-1)
    if my_config.IMAGE_GRADABLE:
        prob_list_gradable, pred_list_gradable, prob_total_gradable, pred_total_gradable, correct_model_no_gradable = \
            predict_single_class(img_file_preprocessed_384, class_type='-3', softmax_or_sigmoids='softmax')

        predict_result['img_gradable'] = pred_total_gradable
        predict_result["img_gradable_0_prob"] = round(prob_total_gradable[0], 2)
        predict_result["img_gradable_1_prob"] = round(prob_total_gradable[1], 2)

        result_s += get_disease_name(predict_result['img_gradable'],
                           class_type='gradable', lang=lang)
        result_s += '\n'

    if my_config.IMAGE_LEFT_RIGHT:
        prob_list_left_right, pred_list_left_right, prob_total_left_right, pred_total_left_right, correct_model_no_left_right = \
            predict_single_class(img_file_preprocessed_384, class_type='-4', softmax_or_sigmoids='softmax')

        predict_result['left_right_eye'] = pred_total_left_right
        predict_result["left_right_eye_0_prob"] = round(prob_total_left_right[0], 2)
        predict_result["left_right_eye_1_prob"] = round(prob_total_left_right[1], 2)


        result_s += get_disease_name(predict_result['left_right_eye'],
                           class_type='left_right', lang=lang)
        result_s += '\n'

    if my_config.IMG_POSITION:
        prob_list_position, pred_list_position, prob_total_position, pred_total_position, correct_model_no_position = \
            predict_single_class(img_file_preprocessed_384, class_type='-5', softmax_or_sigmoids='softmax')

        predict_result['img_position'] = pred_total_position
        predict_result["img_position_0_prob"] = round(prob_total_position[0], 2)
        predict_result["img_position_1_prob"] = round(prob_total_position[1], 2)
        predict_result["img_position_2_prob"] = round(prob_total_position[2], 2)

    if my_config.USE_DIRTY_LENS:
        prob_list_m2, pred_list_m2, prob_total_m2, pred_total_m2, correct_model_no_m2 = \
            predict_single_class(img_file_preprocessed_384, class_type='-2', softmax_or_sigmoids='softmax')

        predict_result['dirty_lens_prob'] = pred_total_m2

    if my_config.USE_OCULAR_SURFACE:  # ocular_surface:0、 fundus image:1、 Others:2
        prob_list_m1, pred_list_m1, prob_total_m1, pred_total_m1, correct_model_no_m1 = \
            predict_single_class(img_file_preprocessed_384, class_type='-1', softmax_or_sigmoids='softmax')

        predict_result['ocular_surface'] = pred_total_m1
        top_n = heapq.nlargest(3, range(len(prob_total_m1)), prob_total_m1.__getitem__)

        predict_result["ocular_surface_0_name"] = get_disease_name(top_n[0], 'm1', lang)
        predict_result["ocular_surface_0_prob"] = str(round(prob_total_m1[top_n[0]] * 100, 1))

        predict_result["ocular_surface_1_name"] = get_disease_name(top_n[1], 'm1', lang)
        predict_result["ocular_surface_1_prob"] = round(prob_total_m1[top_n[1]] * 100, 1)

        predict_result["ocular_surface_2_name"] = get_disease_name(top_n[2], 'm1', lang)
        predict_result["ocular_surface_2_prob"] = round(prob_total_m1[top_n[2]] * 100, 1)

    # endregion

    # region  Big Class
    prob_list_bigclass, pred_list_bigclass, prob_total_bigclass, pred_total_bigclass, correct_model_no_bigclass = \
        predict_single_class(img_file_preprocessed_384, class_type='0', softmax_or_sigmoids=my_config.SOFTMAX_OR_SIGMOIDS)

    predict_result['bigclass_pred'] = pred_total_bigclass
    predict_result['correct_model_no_bigclass'] = correct_model_no_bigclass

    TOP_N_BIG_CLASSES = 3
    top_n = heapq.nlargest(TOP_N_BIG_CLASSES, range(len(prob_total_bigclass)), prob_total_bigclass.__getitem__)

    for i in range(TOP_N_BIG_CLASSES):
        predict_result['bigclass_' + str(i) + '_no'] = top_n[i]  # 大类病种编号
        predict_result['bigclass_' + str(i) + '_name'] = get_disease_name(top_n[i], 'bigclass', lang)
        predict_result['bigclass_' + str(i) + '_prob'] = round(prob_total_bigclass[top_n[i]] * 100, 1)

        result_s += '\n'
        result_s += predict_result['bigclass_' + str(i) + '_name']
        result_s += '\n'
        result_s += str(predict_result['bigclass_' + str(i) + '_prob'])
    result_s += '\n'
    result_s += '\n'

    disease_name += get_disease_name(pred_total_bigclass, 'bigclass', lang)

    # endregion

    # region BigClass CAM and deeplift  bigclass_pred_list only have one value for multi-class
    if cam_type != '-1':
        if pred_total_bigclass not in [0]:  # [0, 29]:
            predict_result['show_cam'] = True

            model_no = correct_model_no_bigclass  # start from 0

            server_port = 23000
            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                # def server_cam(model_no, img_source, pred, cam_relu = True,
                #  preprocess = True, blend_original_image = True

                if cam_type == '0':
                    filename_CAM_orig = proxy1.server_cam(model_no, img_file_preprocessed_384,
                                                          pred_total_bigclass, False, False, True)
                if cam_type == '1':
                    filename_CAM_orig = proxy1.server_cam(model_no, img_file_preprocessed_384,
                                                          pred_total_bigclass, True, False, True)
                if cam_type == '2':
                    # server_gradcam_plusplus(model_no, img_source, pred, preprocess=True,
                    #      blend_original_image=True)
                    filename_CAM_orig = proxy1.server_gradcam_plusplus(model_no, img_file_preprocessed_384,
                                                                       pred_total_bigclass, True)

                # 为了web显示，相对目录
                filename_CAM = os.path.join(baseDir, 'CAM{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM"] = filename_CAM.replace(baseDir, '')

    if show_deeplift:
        if pred_total_bigclass not in [0]:  # [0, 29]:
            predict_result['show_deeplift'] = True

            model_no = correct_model_no_bigclass  # start from 0

            if model_no == 0:
                server_port = 24000
            if model_no == 1:
                server_port = 24001

            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                # server_deep_explain(filename, pred, preprocess)
                filename_CAM_orig = proxy1.server_deep_explain(img_file_preprocessed_384, pred_total_bigclass, False)

                filename_CAM = os.path.join(baseDir, 'CAM_deeplift{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM_deeplift"] = filename_CAM.replace(baseDir, '')

    if show_deepshap:
        if pred_total_bigclass not in [0]:  # [0, 29]:
            # 用哪个模型生成热力图  model_no:1,2,3
            predict_result['show_deepshap'] = True

            model_no = correct_model_no_bigclass  # start from 0

            server_port = 25000
            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                # model_no, img_source, preprocess=True, ranked_outputs=1
                list_classes, list_images = proxy1.server_shap_deep_explain(model_no, img_file_preprocessed_384,
                                                False, 1)

                filename_CAM_orig = list_images[0]

                filename_CAM = os.path.join(baseDir, 'CAM_deepshap{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM_deepshap"] = filename_CAM.replace(baseDir, '')

    # endregion

    # region SubClasses

    disease_name_subclass = {}

    if pred_total_bigclass == 0:
        disease_name_subclass['0'] = ''

        # subclass0  Tigroid fundus
        prob_list_0_1, pred_list_0_1, prob_total_0_1, pred_total_0_1, correct_model_no_0_1 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_1', softmax_or_sigmoids='softmax')

        top_n = heapq.nlargest(2, range(len(prob_total_0_1)), prob_total_0_1.__getitem__)

        if pred_total_0_1 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_1', lang)

            result_s += get_disease_name(top_n[0], 'subclass_0_1', lang)
            result_s += '\n'
            result_s += str(round(prob_total_0_1[top_n[0]] * 100, 1))
            result_s += '\n'

        predict_result["subclass_0_1_1_name"] = get_disease_name(top_n[0], 'subclass_0_1', lang)
        predict_result["subclass_0_1_1_prob"] = round(prob_total_0_1[top_n[0]] * 100, 1)
        predict_result["subclass_0_1_2_name"] = get_disease_name(top_n[1], 'subclass_0_1', lang)
        predict_result["subclass_0_1_2_prob"] = round(prob_total_0_1[top_n[1]] * 100, 1)

        # subclass0 big optic cup
        prob_list_0_2, pred_list_0_2, prob_total_0_2, pred_total_0_2, correct_model_no_0_2 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_2', softmax_or_sigmoids='softmax')

        top_n = heapq.nlargest(2, range(len(prob_total_0_2)), prob_total_0_2.__getitem__)

        if pred_total_0_2 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_2', lang)

            result_s += get_disease_name(top_n[0], 'subclass_0_2', lang)
            result_s += '\n'
            result_s += str(round(prob_total_0_2[top_n[0]] * 100, 1))
            result_s += '\n'

        predict_result["subclass_0_2_1_name"] = get_disease_name(top_n[0], 'subclass_0_2', lang)
        predict_result["subclass_0_2_1_prob"] = round(prob_total_0_2[top_n[0]] * 100, 1)
        predict_result["subclass_0_2_2_name"] = get_disease_name(top_n[1], 'subclass_0_2', lang)
        predict_result["subclass_0_2_2_prob"] = round(prob_total_0_2[top_n[1]] * 100, 1)

        # subclass0  DR1
        prob_list_0_3, pred_list_0_3, prob_total_0_3, pred_total_0_3, correct_model_no_0_3 = \
            predict_single_class(img_file_preprocessed_448, class_type='0_3', softmax_or_sigmoids='softmax')

        top_n = heapq.nlargest(2, range(len(prob_total_0_3)), prob_total_0_3.__getitem__)

        if pred_total_0_3 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_3', lang)

            result_s += get_disease_name(top_n[0], 'subclass_0_3', lang)
            result_s += '\n'
            result_s += str(round(prob_total_0_3[top_n[0]] * 100, 1))
            result_s += '\n'

        predict_result["subclass_0_3_1_name"] = get_disease_name(top_n[0], 'subclass_0_3', lang)
        predict_result["subclass_0_3_1_prob"] = round(prob_total_0_3[top_n[0]] * 100, 1)
        predict_result["subclass_0_3_2_name"] = get_disease_name(top_n[1], 'subclass_0_3', lang)
        predict_result["subclass_0_3_2_prob"] = round(prob_total_0_3[top_n[1]] * 100, 1)

        # DR1用哪个模型生成热力图
        predict_result['subclass_0_3_pred'] = pred_total_0_3
        predict_result["subclass_0_3_correct_no"] = correct_model_no_0_3

        disease_name += ',' + disease_name_subclass['0']

    if pred_total_bigclass in [1, 2, 5, 10, 15, 29]:

        if pred_total_bigclass != 10:
            prob_list_subclass, pred_list_subclass, prob_total_subclass, pred_total_subclass, correct_model_no_subclass = \
                predict_single_class(img_file_preprocessed_384, class_type=str(pred_total_bigclass))
        else:  # subclass10: Probable glaucoma	C/D > 0.7 and Optic atrophy

            found_optic_disc, img_file_crop_optic_disc, img_file_crop_optic_disc_mask = \
                detect_optic_disc(img_file_preprocessed_512, mask=True)
            predict_result['found_optic_disc'] = found_optic_disc

            if found_optic_disc:
                img_file_web_od = os.path.join(baseDir, 'optic_disc_112.jpg')
                shutil.copy(img_file_crop_optic_disc, img_file_web_od)
                predict_result['img_file_crop_optic_disc'] = img_file_web_od

                img_file_web_od_mask = os.path.join(baseDir, 'optic_disc_mask_112.jpg')
                shutil.copy(img_file_crop_optic_disc_mask, img_file_web_od_mask)
                predict_result['img_file_crop_optic_disc_mask'] = img_file_web_od_mask

                # predict SubClasses
                prob_list_subclass, pred_list_subclass, prob_total_subclass, pred_total_subclass, correct_model_no_subclass = \
                    predict_single_class(img_file_web_od, class_type=str(pred_total_bigclass))

            else:
                predict_result['subclass_' + str(pred_total_bigclass) + '_1_name'] = get_disease_name(0,
                      'subclass_' + str(pred_total_bigclass), lang)
                predict_result['subclass_' + str(pred_total_bigclass) + '_1_prob'] = 50
                predict_result['subclass_' + str(pred_total_bigclass) + '_2_name'] = get_disease_name(1,
                        'subclass_' + str(pred_total_bigclass), lang)
                predict_result['subclass_' + str(pred_total_bigclass) + '_2_prob'] = 50

                predict_result['subclass_' + str(pred_total_bigclass) + '_pred'] = 0
                predict_result['subclass_' + str(pred_total_bigclass) + '_correct_no'] = 0

        top_n = heapq.nlargest(2, range(len(prob_total_subclass)), prob_total_subclass.__getitem__)

        disease_name_subclass[str(pred_total_bigclass)] = get_disease_name(top_n[0],
                'subclass_' + str(pred_total_bigclass), lang)

        predict_result['subclass_' + str(pred_total_bigclass) + '_1_name'] = get_disease_name(top_n[0],
                      'subclass_' + str(pred_total_bigclass), lang)
        predict_result['subclass_' + str(pred_total_bigclass) + '_1_prob'] = round(prob_total_subclass[top_n[0]] * 100, 1)
        predict_result['subclass_' + str(pred_total_bigclass) + '_2_name'] = get_disease_name(top_n[1],
                      'subclass_' + str(pred_total_bigclass), lang)
        predict_result['subclass_' + str(pred_total_bigclass) + '_2_prob'] = round(prob_total_subclass[top_n[1]] * 100, 1)

        result_s += '\n'
        result_s += predict_result['subclass_' + str(pred_total_bigclass) + '_1_name']
        result_s += '\n'
        result_s += str(predict_result['subclass_' + str(pred_total_bigclass) + '_1_prob'])
        result_s += '\n'
        result_s += predict_result['subclass_' + str(pred_total_bigclass) + '_2_name']
        result_s += '\n'
        result_s += str(predict_result['subclass_' + str(pred_total_bigclass) + '_2_prob'])
        result_s += '\n'

        # 用哪个模型生成热力图   if big_class_no in [29]:
        predict_result['subclass_' + str(pred_total_bigclass) + '_pred'] = pred_total_subclass
        predict_result['subclass_' + str(pred_total_bigclass) + '_correct_no'] = correct_model_no_subclass

        if predict_result['subclass_{0}_1_prob'.format(pred_total_bigclass)] > \
                predict_result['subclass_{0}_2_prob'.format(pred_total_bigclass)]:
            disease_name += ',' + predict_result['subclass_{0}_1_name'.format(pred_total_bigclass)]
        else:
            disease_name += ',' + predict_result['subclass_{0}_2_name'.format(pred_total_bigclass)]

    predict_result['disease_name'] = disease_name

    # endregion

    # DR lesion areas
    if my_config.LESION_SEG and pred_total_bigclass == 1:
        dict_lesions = predict_dr_lesions(img_file_preprocessed_384)

        img_file_all_lesions = os.path.join(baseDir, checkno,
                                            filename_base + 'all_lesions.jpg')
        os.makedirs(os.path.dirname(filename_CAM), exist_ok=True)
        shutil.copy(dict_lesions['all_lesions'], img_file_all_lesions)
        predict_result['img_file_all_lesions'] = img_file_all_lesions


    pil_save_file = os.path.join(baseDir, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'wb')
    pickle.dump(predict_result, pkl_file)

    file_diagnosis_result = os.path.join(baseDir,  'diagnosis_result.txt')
    fobj = open(file_diagnosis_result, 'w')
    fobj.write(result_s)
    fobj.write('\n')
    fobj.write('\n')
    fobj.write(disease_name)
    fobj.close()

    return


def predict_dr_lesions(img_source):
    server_port = my_config.PORT_BASE + 2000
    SERVER_URL = 'http://localhost:{0}/'.format(server_port)

    with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
        dict_lesions = proxy1.predict_lesions_seg(img_source)

    return dict_lesions

# 分析结果写入数据库，才能查看历史列表
def save_to_db_diagnose(ip, username, image_uuid, diagnostic_results, del_duplicate=False):

    db = db_helper.get_db_conn()
    cursor = db.cursor()

    if del_duplicate:
        if db_helper.DB_TYPE == 'mysql':
            sql = "delete from tb_diagnoses where image_uuid = %s"
        if db_helper.DB_TYPE == 'sqlite':
            sql = "delete from tb_diagnoses where image_uuid = ?"
        cursor.execute(sql, (image_uuid,))
        db.commit()

    if db_helper.DB_TYPE == 'mysql':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(%s,%s,%s,%s) "
    if db_helper.DB_TYPE == 'sqlite':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(?,?,?,?) "
    cursor.execute(sql, (ip, username, image_uuid, diagnostic_results))
    db.commit()

    db.close()


