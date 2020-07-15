
from django.http import HttpResponse
import os, shutil
import xmlrpc.client
import pickle
import json
import my_config

PORT_BASE = my_config.PORT_BASE  #multi_labels: 20000

def get_server_cam_big_class_url():
    port = PORT_BASE + 3000
    url = 'http://localhost:{0}/'.format(port)

    return url

def get_server_deeplift_big_class_url(model_no):
    if model_no == 0:
        port = PORT_BASE + 4000
    if model_no == 1:
        port = PORT_BASE + 4001

    url = 'http://localhost:{0}/'.format(port)

    return url

'''
def get_server_cam_sub_class_url(bigclass):
    if bigclass == '0.2':  #DR1
        url = "http://localhost:23000/"
    if bigclass == '1':  #DR
        url = "http://localhost:23000/"
    if bigclass == '1':  #RVO
        url = "http://localhost:23000/"
    if bigclass == '29': # BLUR
        url = "http://localhost:23000/"

    return url
'''



def get_predict_result(str_uuid):
    # 保存分析结果，以后直接调出
    # web发布目录
    baseDir = os.path.dirname(os.path.abspath(__name__))
    pil_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'rb')
    predict_result = pickle.load(pkl_file)

    return predict_result



def get_saliency_map(request):
    # do not login
    if request.session.get('username', None) is None:
        return HttpResponse('error')

    str_uuid = request.GET.get('str_uuid')
    salienty_type = request.GET.get('type')  #CAM or deeplift

    baseDir = os.path.dirname(os.path.abspath(__name__))
    img_file_preprocessed_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384.jpg')

    predict_result = get_predict_result(str_uuid)

    bigclass_pred_list = predict_result["bigclass_pred_list"]

    dict1 = {}
    top_n_big_classes = 3

    for i in range(top_n_big_classes):
        bigclass_i = predict_result["bigclass_" + str(i) + "_no"]
        if bigclass_i in bigclass_pred_list:
            if bigclass_i not in [0, 29]:

                if salienty_type == 'CAM':
                    model_no = predict_result['list_correct_model_no_bigclass'][bigclass_i]

                    server_url = get_server_cam_big_class_url()

                    with xmlrpc.client.ServerProxy(server_url) as proxy1:
                        # def server_cam(model_no, img_source, pred, cam_relu = True,
                        #  preprocess = True, blend_original_image = True

                        filename_CAM_orig = proxy1.server_cam(model_no, img_file_preprocessed_384,
                                      bigclass_i, True, False, True)

                        # server_gradcam_plusplus(model_no, img_source, pred, preprocess=True,
                        #      blend_original_image=True)
                        # filename_CAM_orig = proxy1.server_gradcam_plusplus(model_no, img_file_preprocessed_384,
                        #       big_class_no,  False, True)

                        # 为了web显示，相对目录
                        filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                                    'CAM_' + str(bigclass_i) + '.jpg')
                        shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                        filename_CAM = filename_CAM.replace(baseDir, '')

                        dict1['filename_CAM' + str(i)] = filename_CAM


                if salienty_type == 'deeplift':

                    model_no = predict_result['list_correct_model_no_bigclass'][bigclass_i]

                    server_url = get_server_deeplift_big_class_url(model_no)

                    with xmlrpc.client.ServerProxy(server_url) as proxy1:

                        filename_CAM_orig = proxy1.server_deep_explain(img_file_preprocessed_384, bigclass_i, False)

                        # 为了web显示，相对目录
                        filename_CAM = os.path.join(baseDir, 'static', 'imgs', str_uuid,
                                                    'deeplift_' + str(bigclass_i) + '.jpg')
                        shutil.copy(filename_CAM_orig, filename_CAM)  # 单个CAM文件copy过来copy过来了
                        filename_CAM = filename_CAM.replace(baseDir, '')

                        dict1['filename_deeplift' + str(i)] = filename_CAM

    return HttpResponse(json.dumps(dict1), content_type="application/json")



