import heapq
import os
import shutil
import xmlrpc.client
import cv2
import importlib
import my_config
from my_module.my_dlp_helper import predict_single_class, get_disease_name, detect_optic_disc, predict_dr_lesions
from my_module.my_image_helper import get_green_channel
from my_module.my_preprocess import my_preprocess


def predict_all_multi_class(str_uuid, file_img_source,  baseDir='/tmp', lang='en',
            cam_type='1', show_deeplift=False, show_deepshap=False):

    importlib.reload(my_config)

    predict_result = {}  # pass dict to view

    predict_result['str_uuid'] = str_uuid
    predict_result['show_cam'] = cam_type
    predict_result['show_deeplift'] = show_deeplift
    predict_result['show_deepshap'] = show_deepshap

    # region image resize, image preprocess 512,448,384
    img_source = cv2.imread(file_img_source)

    # DR1
    img_file_preprocessed_512 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_512.jpg')
    my_preprocess(img_source, crop_size=512,
                  img_file_dest=img_file_preprocessed_512, add_black_pixel_ratio=0)
    img_preprocess_512 = cv2.imread(img_file_preprocessed_512)
    cv2.imwrite(img_file_preprocessed_512, img_preprocess_512)

    img_file_preprocessed_448 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'img_file_preprocessed_448.jpg')
    img_preprocess_448 = cv2.resize(img_preprocess_512, (448, 448))
    cv2.imwrite(img_file_preprocessed_448, img_preprocess_448)
    predict_result['img_file_preprocessed_448'] = img_file_preprocessed_448.replace(baseDir, '')

    # others use 384*384
    IMAGESIZE = 384
    img_file_resized_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'resized_384.jpg')
    img_resized = cv2.resize(img_source, (IMAGESIZE, IMAGESIZE))
    cv2.imwrite(img_file_resized_384, img_resized)
    predict_result['img_file_resized_384'] = img_file_resized_384.replace(baseDir, '')

    img_file_preprocessed_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384.jpg')
    img_preprocess_384 = cv2.resize(img_preprocess_512, (384, 384))
    cv2.imwrite(img_file_preprocessed_384, img_preprocess_384)
    predict_result['img_file_preprocessed_384'] = img_file_preprocessed_384.replace(baseDir, '')

    # endregion

    disease_name = ''

    # region gradable(-3), left_right(-4), position(-5), dirty_lens(-2), ocular_surface(-1)
    if my_config.IMAGE_GRADABLE:
        prob_list_gradable, pred_list_gradable, prob_total_gradable, pred_total_gradable, correct_model_no_gradable = \
            predict_single_class(img_file_preprocessed_384, class_type='-3', softmax_or_sigmoids='softmax')

        predict_result['img_gradable'] = pred_total_gradable
        predict_result["img_gradable_0_prob"] = round(prob_total_gradable[0], 2)
        predict_result["img_gradable_1_prob"] = round(prob_total_gradable[1], 2)

    if my_config.IMAGE_LEFT_RIGHT:
        prob_list_left_right, pred_list_left_right, prob_total_left_right, pred_total_left_right, correct_model_no_left_right = \
            predict_single_class(img_file_preprocessed_384, class_type='-4', softmax_or_sigmoids='softmax')

        predict_result['left_right_eye'] = pred_total_left_right
        predict_result["left_right_eye_0_prob"] = round(prob_total_left_right[0], 2)
        predict_result["left_right_eye_1_prob"] = round(prob_total_left_right[1], 2)

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
            predict_single_class(img_file_resized_384, class_type='-1', softmax_or_sigmoids='softmax')

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
        predict_single_class(img_file_preprocessed_384, class_type='0', softmax_or_sigmoids='softmax')

    predict_result['bigclass_pred'] = pred_total_bigclass
    predict_result['correct_model_no_bigclass'] = correct_model_no_bigclass

    TOP_N_BIG_CLASSES = 3
    #top_n = heapq.nlargest(TOP_N_BIG_CLASSES, range(len(prob_total_bigclass)), prob_total_bigclass.take)
    top_n = heapq.nlargest(TOP_N_BIG_CLASSES, range(len(prob_total_bigclass)), prob_total_bigclass.__getitem__)

    for i in range(TOP_N_BIG_CLASSES):
        predict_result['bigclass_' + str(i) + '_no'] = top_n[i]  # big class no
        predict_result['bigclass_' + str(i) + '_name'] = get_disease_name(top_n[i], 'bigclass', lang)
        predict_result['bigclass_' + str(i) + '_prob'] = round(prob_total_bigclass[top_n[i]] * 100, 1)

    #pred_total_bigclass equal to  top_n[0]
    disease_name += get_disease_name(pred_total_bigclass, 'bigclass', lang)

    # endregion

    # region BigClass CAM and deeplift  bigclass_pred_list only have one value for multi-class
    if cam_type != '-1':
        if pred_total_bigclass not in [0]:  #[0, 29]:
            predict_result['show_cam'] = True

            model_no = correct_model_no_bigclass  # start from 0

            server_port = 23000
            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                if cam_type == '0':
                    # def server_cam(model_no, img_source, pred, cam_relu = True,
                    #  preprocess = True, blend_original_image = True
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
                filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                                'CAM{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM"] = filename_CAM.replace(baseDir, '')

    if show_deeplift:
        if pred_total_bigclass not in [0]: #[0, 29]:
            predict_result['show_deeplift'] = True

            model_no = correct_model_no_bigclass   # start from 0

            if model_no == 0:
                server_port = 24000
            if model_no == 1:
                server_port = 24001

            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                # server_deep_explain(filename, pred, preprocess)
                filename_CAM_orig = proxy1.server_deep_explain(img_file_preprocessed_384, pred_total_bigclass, False)

                filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                    'CAM_deeplift{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM_deeplift"] = filename_CAM.replace(baseDir, '')

    if show_deepshap:
        if pred_total_bigclass not in [0]:  #[0, 29]:
            predict_result['show_deepshap'] = True

            model_no = correct_model_no_bigclass  # start from 0

            server_port = 25000
            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                #model_no, img_source, preprocess=True, ranked_outputs=1
                list_classes, list_images = proxy1.server_shap_deep_explain(model_no, img_file_preprocessed_384,
                      False, 1)

                filename_CAM_orig = list_images[0]
                filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                    'CAM_deepshap{}.jpg'.format(pred_total_bigclass))

                shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                filename_CAM = filename_CAM.replace(baseDir, '')

                predict_result["bigclass_0_CAM_deepshap"] = filename_CAM.replace(baseDir, '')

    # endregion

    # region SubClasses
    disease_name_subclass = {}  # can support both multi-class and mulit-label

    if pred_total_bigclass == 0:
        disease_name += '  ('

        disease_name_subclass['0'] = ''

        # subclass0  Tigroid fundus
        prob_list_0_1, pred_list_0_1, prob_total_0_1, pred_total_0_1, correct_model_no_0_1 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_1', softmax_or_sigmoids='softmax')

        top_n = heapq.nlargest(2, range(len(prob_total_0_1)), prob_total_0_1.__getitem__)

        if pred_total_0_1 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_1', lang) + ', '

        # predict_result["subclass_0_1_1_name"] = get_disease_name(top_n[0], 'subclass_0_1', lang)
        # predict_result["subclass_0_1_1_prob"] = round(prob_total_0_1[top_n[0]] * 100, 1)
        # predict_result["subclass_0_1_2_name"] = get_disease_name(top_n[1], 'subclass_0_1', lang)
        # predict_result["subclass_0_1_2_prob"] = round(prob_total_0_1[top_n[1]] * 100, 1)

        predict_result["subclass_0_1_1_name"] = get_disease_name(0, 'subclass_0_1', lang)
        predict_result["subclass_0_1_1_prob"] = round(prob_total_0_1[0] * 100, 1)
        predict_result["subclass_0_1_2_name"] = get_disease_name(1, 'subclass_0_1', lang)
        predict_result["subclass_0_1_2_prob"] = round(prob_total_0_1[1] * 100, 1)

        # subclass0 big optic cup
        prob_list_0_2, pred_list_0_2, prob_total_0_2, pred_total_0_2, correct_model_no_0_2 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_2', softmax_or_sigmoids='softmax')

        top_n = heapq.nlargest(2, range(len(prob_total_0_2)), prob_total_0_2.__getitem__)

        if pred_total_0_2 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_2', lang) + ', '

        # predict_result["subclass_0_2_1_name"] = get_disease_name(top_n[0], 'subclass_0_2', lang)
        # predict_result["subclass_0_2_1_prob"] = round(prob_total_0_2[top_n[0]] * 100, 1)
        # predict_result["subclass_0_2_2_name"] = get_disease_name(top_n[1], 'subclass_0_2', lang)
        # predict_result["subclass_0_2_2_prob"] = round(prob_total_0_2[top_n[1]] * 100, 1)

        predict_result["subclass_0_2_1_name"] = get_disease_name(0, 'subclass_0_2', lang)
        predict_result["subclass_0_2_1_prob"] = round(prob_total_0_2[0] * 100, 1)
        predict_result["subclass_0_2_2_name"] = get_disease_name(1, 'subclass_0_2', lang)
        predict_result["subclass_0_2_2_prob"] = round(prob_total_0_2[1] * 100, 1)

        # subclass0  DR1
        prob_list_0_3, pred_list_0_3, prob_total_0_3, pred_total_0_3, correct_model_no_0_3 = \
            predict_single_class(img_file_preprocessed_448, class_type='0_3', softmax_or_sigmoids='softmax')

        if prob_total_0_3[1] > 0.15:
            prob_total_0_3[0] += 0.15
            prob_total_0_3[1] -= 0.15
            if prob_total_0_3[0] >= prob_total_0_3[1]:
                pred_total_0_3 = 0
            else:
                pred_total_0_3 = 1

        top_n = heapq.nlargest(2, range(len(prob_total_0_3)), prob_total_0_3.__getitem__)

        if pred_total_0_3 == 1:
            disease_name_subclass['0'] += get_disease_name(top_n[0], 'subclass_0_3', lang) + ', '

        # predict_result["subclass_0_3_1_name"] = get_disease_name(top_n[0], 'subclass_0_3', lang)
        # predict_result["subclass_0_3_1_prob"] = round(prob_total_0_3[top_n[0]] * 100, 1)
        # predict_result["subclass_0_3_2_name"] = get_disease_name(top_n[1], 'subclass_0_3', lang)
        # predict_result["subclass_0_3_2_prob"] = round(prob_total_0_3[top_n[1]] * 100, 1)

        predict_result["subclass_0_3_1_name"] = get_disease_name(0, 'subclass_0_3', lang)
        predict_result["subclass_0_3_1_prob"] = round(prob_total_0_3[0] * 100, 1)
        predict_result["subclass_0_3_2_name"] = get_disease_name(1, 'subclass_0_3', lang)
        predict_result["subclass_0_3_2_prob"] = round(prob_total_0_3[1] * 100, 1)

        # DR1用哪个模型生成热力图
        predict_result['subclass_0_3_pred'] = pred_total_0_3
        predict_result["subclass_0_3_correct_no"] = correct_model_no_0_3

        if pred_total_0_1 == 0 and pred_total_0_2 == 0 and pred_total_0_3 == 0:
            disease_name += 'Normal'
        else:
            disease_name += disease_name_subclass['0'][:-2] #delete last black space

        disease_name += ')'

    if pred_total_bigclass in [1, 2, 5, 10, 15, 29]:
        disease_name += '  ('

        if pred_total_bigclass not in [1, 10]:
            prob_list_i, pred_list_i, prob_total_i, pred_total_i, correct_model_no_i = \
                predict_single_class(img_file_preprocessed_384, class_type=str(pred_total_bigclass))

        elif pred_total_bigclass == 1: #DR2,3, ensembling Neovascularization
            prob_list_i, pred_list_i, prob_total_i, pred_total_i, correct_model_no_i = \
                predict_single_class(img_file_preprocessed_384, class_type=str(pred_total_bigclass))

            img_file_preprocessed_384_G = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384_G.jpg')
            get_green_channel(img_file_preprocessed_384, img_file_preprocessed_384_G)

            prob_list_neo, pred_list_neo, prob_total_neo, pred_total_neo, correct_model_no_neo = \
                predict_single_class(img_file_preprocessed_384_G, class_type='60')

            if pred_total_neo == 1 and pred_total_i == 0:
                prob_total_i = prob_total_neo
                pred_total_i = pred_total_neo
                correct_model_no_i = correct_model_no_neo

        else:  # subclass10: Probable glaucoma	C/D > 0.7 and Optic atrophy
            # detect optic disc
            found_optic_disc, img_file_crop_optic_disc, img_file_crop_optic_disc_mask = \
                detect_optic_disc(img_file_preprocessed_512, mask=True)
            predict_result['found_optic_disc'] = found_optic_disc

            if found_optic_disc:
                img_file_web_od = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'optic_disc_112.jpg')
                shutil.copy(img_file_crop_optic_disc, img_file_web_od)
                predict_result['img_file_crop_optic_disc'] = img_file_web_od.replace(baseDir, '')

                img_file_web_od_mask = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'optic_disc_mask_112.jpg')
                shutil.copy(img_file_crop_optic_disc_mask, img_file_web_od_mask)
                predict_result['img_file_crop_optic_disc_mask'] = img_file_web_od_mask.replace(baseDir, '')

                # predict SubClasses
                prob_list_i, pred_list_i, prob_total_i, pred_total_i, correct_model_no_i = \
                    predict_single_class(img_file_web_od, class_type=str(pred_total_bigclass))

            else:

                predict_result['subclass_' + str(pred_total_bigclass) + '_1_name'] = get_disease_name(0,
                                                           'subclass_' + str( pred_total_bigclass), lang)
                predict_result['subclass_' + str(pred_total_bigclass) + '_1_prob'] = 50
                predict_result['subclass_' + str(pred_total_bigclass) + '_2_name'] = get_disease_name(1,
                                                            'subclass_' + str(pred_total_bigclass), lang)
                predict_result['subclass_' + str(pred_total_bigclass) + '_2_prob'] = 50

                predict_result['subclass_' + str(pred_total_bigclass) + '_pred'] = 0
                predict_result['subclass_' + str(pred_total_bigclass) + '_correct_no'] = 0

        top_n = heapq.nlargest(2, range(len(prob_total_i)), prob_total_i.__getitem__)

        disease_name_subclass[str(pred_total_bigclass)] = get_disease_name(top_n[0], 'subclass_' + str(pred_total_bigclass),
                                                                    lang)

        predict_result['subclass_' + str(pred_total_bigclass) + '_1_name'] = get_disease_name(top_n[0],
                                       'subclass_' + str(pred_total_bigclass), lang)
        predict_result['subclass_' + str(pred_total_bigclass) + '_1_prob'] = round(prob_total_i[top_n[0]] * 100, 1)
        predict_result['subclass_' + str(pred_total_bigclass) + '_2_name'] = get_disease_name(top_n[1],
                                       'subclass_' + str(pred_total_bigclass), lang)
        predict_result['subclass_' + str(pred_total_bigclass) + '_2_prob'] = round(prob_total_i[top_n[1]] * 100, 1)

        # 用哪个模型生成热力图   if big_class_no in [29]:
        predict_result['subclass_' + str(pred_total_bigclass) + '_pred'] = pred_total_i
        predict_result['subclass_' + str(pred_total_bigclass) + '_correct_no'] = correct_model_no_i

        if prob_total_i[top_n[0]] > \
                prob_total_i[top_n[1]]:
            disease_name += predict_result['subclass_{0}_1_name'.format(pred_total_bigclass)]
        else:
            disease_name += predict_result['subclass_{0}_2_name'.format(pred_total_bigclass)]

        disease_name += ')'

    predict_result['disease_name'] = disease_name

    # endregion

    # DR lesion areas
    if my_config.LESION_SEG and pred_total_bigclass == 1:
        dict_lesions = predict_dr_lesions(img_file_preprocessed_384)
        img_file_all_lesions = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'all_lesions.jpg')
        shutil.copy(dict_lesions['all_lesions'], img_file_all_lesions)
        predict_result['img_file_all_lesions'] = img_file_all_lesions.replace(baseDir, '')


    return predict_result