import copy
import heapq
import importlib
import os
import shutil
import xmlrpc.client
import cv2
import my_config
from my_module.my_dlp_helper import predict_single_class, get_disease_name, get_server_cam_big_class_multi_label_url, \
    get_deeplift_big_class_url, detect_optic_disc, predict_dr_lesions
from my_module.my_preprocess import my_preprocess


def predict_all_multi_labels(str_uuid, file_img_source, baseDir, lang,
            cam_type='1', show_deeplift=False, show_deepshap=False):

    importlib.reload(my_config)

    predict_result = {}     #传递给view
    predict_result['str_uuid'] = str_uuid
    predict_result['show_cam'] = cam_type

    #region resize and preprocess
    img_source = cv2.imread(file_img_source)

    img_file_resized_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'resized_384.jpg')
    IMAGE_SIZE = 384
    img_resized = cv2.resize(img_source, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imwrite(img_file_resized_384, img_resized)
    predict_result['img_file_resized_384'] = img_file_resized_384

    img_file_preprocessed_512 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_512.jpg')
    img_file_preprocessed_448 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_448.jpg')
    img_file_preprocessed_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384.jpg')

    my_preprocess(img_source, crop_size=512,
                  img_file_dest=img_file_preprocessed_512,
                  add_black_pixel_ratio=my_config.ADD_BLACK_PIXEL_RATIO)
    img_preprocess_512 = cv2.imread(img_file_preprocessed_512)
    cv2.imwrite(img_file_preprocessed_512, img_preprocess_512)

    img_preprocess_384 = cv2.resize(img_preprocess_512, (384, 384))
    cv2.imwrite(img_file_preprocessed_384, img_preprocess_384)
    predict_result['img_file_preprocessed_384'] = img_file_preprocessed_384

    #endregion

    #region gradable, left_right, position, dirty_lens, ocular_surface
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

    if my_config.USE_OCULAR_SURFACE:  # 检测眼底:0、眼表:1、其他:2 没有预处理
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

    #endregion

    #region  Big Class, multi-label
    prob_list_bigclass, pred_list_bigclass, prob_total_bigclass, pred_total_bigclass, list_correct_model_no_bigclass = \
        predict_single_class(img_file_preprocessed_384, class_type='0_10', softmax_or_sigmoids='sigmoid')

    from my_module.my_multilabel_depedence import postprocess_exclusion, postprocess_all_negative, postprocess_multi_positive
    if my_config.POSTPROCESS_EXCLUSION:
        prob_total_bigclass = postprocess_exclusion(prob_total_bigclass, my_config.LIST_THRESHOLD)
    if my_config.POSTPROCESS_ALLNEGATIVE:
        prob_total_bigclass = postprocess_all_negative(prob_total_bigclass, my_config.LIST_THRESHOLD)
    if my_config.POSTPROCESS_MULTI_POSITIVE:
        prob_total_bigclass = postprocess_multi_positive(prob_total_bigclass, my_config.LIST_THRESHOLD)

    #recalculate pred_total_bigclass, list_correct_model_no_bigclass
    list_bigclass = []
    for i in range(my_config.NUM_CLASSES):
        if prob_total_bigclass[i] > my_config.LIST_THRESHOLD[i]:
            list_bigclass.append(i)

    list_bigclass_model_no = []
    for i in range(my_config.NUM_CLASSES):
        for j in range(len(pred_list_bigclass)):
            if pred_list_bigclass[j][i] > my_config.LIST_THRESHOLD[i]:
                list_bigclass_model_no.append(j)
                break

    # big classes, list of positive diseases(exclude non-referable)
    list_bigclass_non_referable = copy.copy(list_bigclass)
    if 0 in list_bigclass_non_referable:
        list_bigclass_non_referable.remove(0)

    predict_result["bigclass_pred_list"] = list_bigclass_non_referable  # (2,29)

    disease_name_bigclass = {}
    if len(list_bigclass_non_referable) == 0:    # Non-referable
        disease_name_bigclass['0'] = get_disease_name(0, 'bigclass', lang)

    #排除正常大类的概率    获取大类病种概率top n,
    prob_total_bigclass[0] = 0

    top_n_big_classes = 3
    top_n = heapq.nlargest(top_n_big_classes, range(len(prob_total_bigclass)), prob_total_bigclass.__getitem__)

    for i in range(top_n_big_classes):
        if pred_total_bigclass[top_n[i]] == 1:
            disease_name_bigclass[str(top_n[i])] = get_disease_name(top_n[i], 'bigclass', lang)

        predict_result['bigclass_' + str(i) + '_no'] = top_n[i]  #大类病种编号
        predict_result['bigclass_' + str(i) + '_name'] = get_disease_name(top_n[i], 'bigclass', lang)
        predict_result['bigclass_' + str(i) + '_prob'] = round(prob_total_bigclass[top_n[i]] * 100, 1)

    #endregion

    # region bigclass saliency maps
    for big_class_no in list_bigclass_non_referable:
        if cam_type != '-1':
            if big_class_no not in [0]:
                # 用哪个模型生成热力图  model_no:1,2,3
                model_no = list_correct_model_no_bigclass[big_class_no]  # start from 0

                with xmlrpc.client.ServerProxy(get_server_cam_big_class_multi_label_url()) as proxy1:
                    # def server_cam(model_no, img_source, pred, cam_relu = True,
                    #  preprocess = True, blend_original_image = True

                    # 0:Original CAM, 1:Modified CAM, 2:Grad CAM++
                    if cam_type == '0':
                        filename_CAM_orig = proxy1.server_cam(model_no, img_file_preprocessed_384,
                                                          big_class_no, False, False, True)
                    if cam_type == '1':
                        filename_CAM_orig = proxy1.server_cam(model_no, img_file_preprocessed_384,
                                                          big_class_no, True, False, True)
                    if cam_type == '2':
                        # server_gradcam_plusplus(model_no, img_source, pred, preprocess=True,
                        #      blend_original_image=True)
                        filename_CAM_orig = proxy1.server_gradcam_plusplus(model_no, img_file_preprocessed_384,
                              big_class_no,  False, True)

                    # 为了web显示，相对目录
                    filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                                'CAM' + str(big_class_no) + '.jpg')
                    shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                    filename_CAM = filename_CAM.replace(baseDir, '')

                    # predict_result["CAM" + str(big_class_no)] = filename_CAM.replace(baseDir, '')

                    if big_class_no == top_n[0]:
                        predict_result["CAM_index_0"] = filename_CAM.replace(baseDir, '')
                    if big_class_no == top_n[1]:
                        predict_result["CAM_index_1"] = filename_CAM.replace(baseDir, '')
                    if big_class_no == top_n[2]:
                        predict_result["CAM_index_2"] = filename_CAM.replace(baseDir, '')

        if show_deeplift:
            # 用哪个模型生成热力图  model_no: start from 0
            model_no = list_correct_model_no_bigclass[big_class_no]

            with xmlrpc.client.ServerProxy(get_deeplift_big_class_url(model_no)) as proxy1:
                # server_deep_explain(filename, pred, preprocess)
                filename_CAM_orig = proxy1.server_deep_explain(img_file_preprocessed_384, big_class_no, False)

                # 为了web显示，相对目录
                filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                            'CAM_deeplift' + str(big_class_no) + '.jpg')
                shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                filename_CAM = filename_CAM.replace(baseDir, '')

                # predict_result["CAM_deeplift" + str(big_class_no)] = filename_CAM.replace(baseDir, '')

                if big_class_no == top_n[0]:
                    predict_result["CAM_deeplift_0"] = filename_CAM.replace(baseDir, '')
                if big_class_no == top_n[1]:
                    predict_result["CAM_deeplift_1"] = filename_CAM.replace(baseDir, '')
                if big_class_no == top_n[2]:
                    predict_result["CAM_deeplift_2"] = filename_CAM.replace(baseDir, '')

    if show_deepshap:
        bigclass_pred_list_multi_model = []
        for pred_classes_single_model in pred_list_bigclass:
            list1 = []
            for big_class_no in range(1, len(pred_classes_single_model)):
                if pred_classes_single_model[big_class_no] == 1:
                    list1.append(big_class_no)

            bigclass_pred_list_multi_model.append(list1)

        if len(list_bigclass_non_referable) > 1 or\
                (len(list_bigclass_non_referable) == 1 and (0 not in list_bigclass_non_referable) and
                 (29 not in list_bigclass_non_referable)):

            predict_result["show_deepshap"] = True

            server_port = my_config.PORT_DEEP_SHAP
            SERVER_URL = 'http://localhost:{0}/'.format(server_port)

            #xception first
            if set(list_bigclass_non_referable) <= set(bigclass_pred_list_multi_model[1]):
                with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                    list_classes, list_images = proxy1.server_shap_deep_explain(1,
                            img_file_preprocessed_384, False,
                                len(bigclass_pred_list_multi_model[1]))

            elif set(list_bigclass_non_referable) <= set(bigclass_pred_list_multi_model[0]):
                with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                    list_classes, list_images = proxy1.server_shap_deep_explain(0,
                                    img_file_preprocessed_384, False,
                                    len(bigclass_pred_list_multi_model[0]))

            elif set(list_bigclass_non_referable) <= set(bigclass_pred_list_multi_model[2]):
                with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                    list_classes, list_images = proxy1.server_shap_deep_explain(2,
                                    img_file_preprocessed_384, False,
                                    len(bigclass_pred_list_multi_model[2]))

            else:
                with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
                    list_classes, list_images = proxy1.server_shap_deep_explain(1,
                                    img_file_preprocessed_384, False,
                                    len(bigclass_pred_list_multi_model[1]))


            for j, class1 in enumerate(list_classes):
                # 为了web显示，相对目录
                filename_deepshap = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                                 'deepshap' + str(class1) + '.jpg')

                shutil.copy(list_images[j], filename_deepshap)
                filename_deepshap = filename_deepshap.replace(baseDir, '')

                if class1 == top_n[0]:
                    predict_result["Deepshap_index_0"] = filename_deepshap.replace(baseDir, '')
                if class1 == top_n[1]:
                    predict_result["Deepshap_index_1"] = filename_deepshap.replace(baseDir, '')
                if class1 == top_n[2]:
                    predict_result["Deepshap_index_2"] = filename_deepshap.replace(baseDir, '')

    # endregion

    #region SubClasses
    list_subclass = []

    disease_name_subclass = {}
    # Non-referable because all disease classes are negative, special, 3-subclass
    if len(list_bigclass_non_referable) == 0:
        disease_name_subclass['0'] = ''
        predict_result["bigclass_normal"] = 1

        # subclass0_1, Tigroid fundus
        prob_list_0_1, pred_list_0_1, prob_total_0_1, pred_total_0_1, correct_model_no_0_1 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_1', softmax_or_sigmoids='softmax')
        if pred_total_0_1 == 1:
            disease_name_subclass['0'] += get_disease_name(pred_total_0_1, 'subclass_0_1', lang)
            list_subclass.append(('0_1', 1))
        predict_result["subclass_0_1_1_name"] = get_disease_name(0, 'subclass_0_1', lang)
        predict_result["subclass_0_1_1_prob"] = round(prob_total_0_1[0] * 100, 1)
        predict_result["subclass_0_1_2_name"] = get_disease_name(1, 'subclass_0_1', lang)
        predict_result["subclass_0_1_2_prob"] = round(prob_total_0_1[1] * 100, 1)

        # subclass0_2  big optic cup
        prob_list_0_2, pred_list_0_2, prob_total_0_2, pred_total_0_2, correct_model_no_0_2 = \
            predict_single_class(img_file_preprocessed_384, class_type='0_2', softmax_or_sigmoids='softmax')
        if pred_total_0_2 == 1:
            disease_name_subclass['0'] += get_disease_name(pred_total_0_2, 'subclass_0_2', lang)
            list_subclass.append(('0_2', 1))
        predict_result["subclass_0_2_1_name"] = get_disease_name(0, 'subclass_0_2', lang)
        predict_result["subclass_0_2_1_prob"] = round(prob_total_0_2[0] * 100, 1)
        predict_result["subclass_0_2_2_name"] = get_disease_name(1, 'subclass_0_2', lang)
        predict_result["subclass_0_2_2_prob"] = round(prob_total_0_2[1] * 100, 1)

        # subclass0_3  Normal, DR1
        if my_config.DR1:
            img_preprocess_448 = cv2.resize(img_preprocess_512, (448, 448))
            cv2.imwrite(img_file_preprocessed_448, img_preprocess_448)
            predict_result['img_file_preprocessed_448'] = img_file_preprocessed_448

            prob_list_0_3, pred_list_0_3, prob_total_0_3, pred_total_0_3, correct_model_no_0_3 = \
                predict_single_class(img_file_preprocessed_448, class_type='0_3', softmax_or_sigmoids='softmax')
            if pred_total_0_3 == 1:
                disease_name_subclass['0'] += get_disease_name(pred_total_0_3, 'subclass_0_3', lang)
                list_subclass.append(('0_3', 1))
            predict_result["subclass_0_3_1_name"] = get_disease_name(top_n[0], 'subclass_0_3', lang)
            predict_result["subclass_0_3_1_prob"] = round(prob_total_0_3[top_n[0]] * 100, 1)
            predict_result["subclass_0_3_2_name"] = get_disease_name(top_n[1], 'subclass_0_3', lang)
            predict_result["subclass_0_3_2_prob"] = round(prob_total_0_3[top_n[1]] * 100, 1)

            # 用哪个模型生成热力图
            predict_result['subclass_0_3_pred'] = pred_total_0_3
            predict_result["subclass_0_3_correct_no"] = correct_model_no_0_3

    else:
        for big_class_no in list_bigclass_non_referable:
            if big_class_no in [1, 2, 5, 10, 15, 29]:
                if big_class_no == 10:
                    # subclass11 --  Probable glaucoma	C/D > 0.7 and Optic atrophy	pale with normal cupping
                    found_optic_disc, img_file_crop_optic_disc, img_file_crop_optic_disc_mask = \
                        detect_optic_disc(img_file_preprocessed_512, mask=True)
                    predict_result['found_optic_disc'] = found_optic_disc

                    if found_optic_disc:
                        img_file_web_od = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'optic_disc_112.jpg')
                        shutil.copy(img_file_crop_optic_disc, img_file_web_od)
                        predict_result['img_file_crop_optic_disc'] = img_file_web_od

                        img_file_web_od_mask = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'optic_disc_mask_112.jpg')
                        shutil.copy(img_file_crop_optic_disc_mask, img_file_web_od_mask)
                        predict_result['img_file_crop_optic_disc_mask'] = img_file_web_od_mask

                        #predict SubClasses
                        prob_list_i, pred_list_i, prob_total_i, pred_total_i, correct_model_no_i = \
                        predict_single_class(img_file_web_od, class_type=str(big_class_no))

                    else:
                        disease_name_subclass[str(big_class_no)] = get_disease_name(0, 'subclass_' + str(big_class_no), lang)

                        predict_result['subclass_' + str(big_class_no) + '_1_name'] = get_disease_name(0, 'subclass_' + str(big_class_no), lang)
                        predict_result['subclass_' + str(big_class_no) + '_1_prob'] = 50
                        predict_result['subclass_' + str(big_class_no) + '_2_name'] = get_disease_name(1, 'subclass_' + str(big_class_no), lang)
                        predict_result['subclass_' + str(big_class_no) + '_2_prob'] = 50

                        # 用哪个模型生成热力图       if big_class_no in [28]:
                        predict_result['subclass_' + str(big_class_no) + '_pred'] = 0
                        predict_result['subclass_' + str(big_class_no) + '_correct_no'] = correct_model_no_i

                else:
                    prob_list_i, pred_list_i, prob_total_i, pred_total_i, correct_model_no_i = \
                        predict_single_class(img_file_preprocessed_384, class_type=str(big_class_no))

                #subclasses using the default threshold
                list_subclass.append((str(big_class_no), pred_total_i))

                top_n = heapq.nlargest(2, range(len(prob_total_i)), prob_total_i.__getitem__)
                disease_name_subclass[str(big_class_no)] = get_disease_name(top_n[0], 'subclass_' + str(big_class_no), lang)

                predict_result['subclass_' + str(big_class_no) + '_1_name'] = get_disease_name(top_n[0], 'subclass_' + str(big_class_no), lang)
                predict_result['subclass_' + str(big_class_no) + '_1_prob'] = round(prob_total_i[top_n[0]] * 100, 1)
                predict_result['subclass_' + str(big_class_no) + '_2_name'] = get_disease_name(top_n[1], 'subclass_' + str(big_class_no), lang)
                predict_result['subclass_' + str(big_class_no) + '_2_prob'] = round(prob_total_i[top_n[1]] * 100, 1)

                # 用哪个模型生成热力图       if big_class_no in [28]:
                predict_result['subclass_' + str(big_class_no) + '_pred'] = pred_total_i
                predict_result['subclass_' + str(big_class_no) + '_correct_no'] = correct_model_no_i

    # endregion

    #region DR lesion areas
    if my_config.LESION_SEG and [1] in list_bigclass_non_referable:
        dict_lesions = predict_dr_lesions(img_file_preprocessed_384)

        img_file_all_lesions = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'all_lesions.jpg')
        shutil.copy(dict_lesions['all_lesions'], img_file_all_lesions)
        predict_result['img_file_all_lesions'] = img_file_all_lesions

    # endregion

    #region disease_name
    disease_name = ''
    for key, value in disease_name_bigclass.items():
        disease_name += value

        if key in disease_name_subclass:
            if disease_name_subclass[key] != '':
                disease_name += '(' + disease_name_subclass[key] + '),'

        if not disease_name.endswith(','):
            disease_name += ','

    if disease_name.endswith(','):
        disease_name = disease_name[:-1]

    predict_result['disease_name'] = disease_name
    #region

    #region  ent levelurg
    #urgence  O observation, R routine, S semi-urgent, U urgent
    urgence_level = 0
    for bigclass in list_bigclass:
        #Routine
        if bigclass in [13, 14, 15, 17, 27, 28]:
            if urgence_level < 1:
                urgence_level = 1
        #Semi-urgent
        if bigclass in [2, 7, 9, 16, 20, 21, 22, 23, 24]:
            if urgence_level < 2:
                urgence_level = 2
        #Urgent
        if bigclass in [3, 4, 6, 8, 11, 12, 19, 25, 26]:
            if urgence_level < 3:
                urgence_level = 3

    #Subclass
    '''
    subclass_0_1_1_prob, subclass_0_2_1_prob, subclass_0_3_1_prob
    predict_result['subclass_' + str(big_class_no) + '_1_prob']
    0.0 O, 0.1 O, 0.2 R, 0.3 R
    1.0 S, 1.1 U
    5.0 S, 5.1 U
    10.1 S, 10.1 U
    29.0 S, 29.1 U
    '''
    for (big, sub) in list_subclass:
        if big in ['0_2', '0_3'] and sub == 1:
            if urgence_level < 1:
                urgence_level = 1

        if big in ['1', '5', '29']:
            if sub == 0:
                if urgence_level < 2:
                    urgence_level = 2
            if sub == 1:
                if urgence_level < 3:
                    urgence_level = 3

        if big in ['10']:
            if sub == 0:
                if urgence_level < 3:
                    urgence_level = 3
            if sub == 1:
                if urgence_level < 2:
                    urgence_level = 2

    if lang == 'en':
        list_urgence_level = ['observation', 'routine', 'semi-urgent', 'urgent']
    else:
        list_urgence_level = ['观察', '常规检查', '比较紧急', '紧急']
    predict_result['urgence_level'] = list_urgence_level[urgence_level]

    #endregion

    # 修改相对目录 网页显示图像, 目录
    for img_str in ['img_file_resized_384', 'img_file_preprocessed_384',
                    'img_file_preprocessed_512', 'img_file_preprocessed_448',
                    'img_file_crop_optic_disc', 'img_file_crop_optic_disc_mask']:
        if img_str in predict_result:
            if predict_result[img_str] != '':
                predict_result[img_str] = predict_result[img_str].replace(baseDir, '')

    # 返回结果集
    return predict_result


