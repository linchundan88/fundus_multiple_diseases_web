'''

diagnose 接受上传的文件，进行分析，分析结果调用模版diagnosis.html
predict_result 构造给view的数据结构，也是字典

'''

import os
from django.shortcuts import render
import uuid
import pickle
import cv2

from my_module import my_dlp_helper

# accept http parameters, call my_dlp_helper, save results to pkl, call view
def diagnose(request):
    if request.session.get('username', None) is None:
        return render(request, 'homepage.html')

    #region request parameters, receive the uploaded fundus image

    lang = str(request.session['lang'])

    ## 0:CAM, 1:Cam With Relu , 2: Grad-Cam++
    cam_type = request.POST.get('cam_type', '1')

    # show_deeplift = (request.POST.get('show_deeplift') == '1')
    show_deeplift = False

    show_deepshap = (request.POST.get('show_deepshap') == '1')

    if request.FILES.get('input_fundus_image') == None:
        if lang == 'en':
            file_view = 'error.html'
            err_msg = "Please choose an image, and then click button 'Analyse'"
        else:
            file_view = 'error_cn.html'
            err_msg = "清先选择文件，然后点击分析按钮。"
        return render(request, file_view, {'err_msg': err_msg})

    webDir = os.path.dirname(os.path.abspath(__name__))

    str_uuid = str(uuid.uuid1())  # generate an unique ID
    request.session['image_uuid'] = str_uuid  #feedback will use

    request_filename = request.FILES.get('input_fundus_image')
    img_dir = os.path.join(webDir, 'static', 'imgs', str_uuid)
    os.makedirs(img_dir, exist_ok=True)

    filename_save = os.path.join(img_dir, request_filename.name)
    fobj = open(filename_save, 'wb')
    for chrunk in request_filename.chunks():
        fobj.write(chrunk)
    fobj.close()

    #endregion

    #region image validation,  save to original.jpg avoid image file confliction(very low probability)
    try:
        img_temp = cv2.imread(filename_save)
        cv2.imwrite(os.path.join(img_dir, 'original.jpg'), img_temp)

        (height, width, channel) = img_temp.shape
        if channel not in [3, 4]:
            if lang == 'en':
                err_msg = "image channel number error. The image must be color image."
                file_view = 'error.html'
            else:
                err_msg = "图像通道数错误,图像必须是彩色图像。"
                file_view = 'error_cn.html'
            return render(request, file_view, {'err_msg': err_msg})
        else:
            if width < 384:
                if lang == 'en':
                    err_msg = "The width of the image must be larger than 384, current width value:{} ".format(width)
                    file_view = 'error.html'
                else:
                    err_msg = "图像的宽必须大于384。"
                    file_view = 'error_cn.html'
                return render(request, file_view, {'err_msg': err_msg})

            if height < 384:
                if lang == 'en':
                    err_msg = "The height of the image must be larger than 384, current height value:{} ".format(height)
                    file_view = 'error.html'
                else:
                    err_msg = "图像的高必须大于384。"
                    file_view = 'error_cn.html'
                return render(request, file_view, {'err_msg': err_msg})
    except:
        if lang == 'en':
            err_msg = "Invalid image file"
            file_view = 'error.html'
        else:
            err_msg = "无效的图像文件。"
            file_view = 'error_cn.html'
        return render(request, file_view, {'err_msg': err_msg})

    # endregion

    #region do predict
    if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
        predict_result = my_dlp_helper.predict_all_multi_class(str_uuid, filename_save, webDir, lang,
               cam_type= cam_type,
               show_deeplift=show_deeplift, show_deepshap=show_deepshap)
    else:
        predict_result = my_dlp_helper.predict_all_multi_labels(str_uuid, filename_save, webDir, lang,
                cam_type=cam_type,
                show_deeplift=show_deeplift, show_deepshap=show_deepshap)

    #endregion

    #region save result to pkl file and database.
    pil_save_file = os.path.join(webDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'wb')
    pickle.dump(predict_result, pkl_file)

    username = request.session.get('username')
    diagnostic_results = predict_result['disease_name']

    if 'HTTP_X_FORWARDED_FOR' in request.META:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.META['REMOTE_ADDR']

    my_dlp_helper.save_to_db_diagnose(ip, username, str_uuid, diagnostic_results)

    #endregion

    if lang == 'en':
        if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
            file_view = 'diagnosis_mc.html'
        else:
            file_view = 'diagnosis_ml.html'
    else:
        if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
            file_view = 'diagnosis_mc_cn.html'
        else:
            file_view = 'diagnosis_ml_cn.html'

    return render(request, file_view,  {'realtime_diagnose': True, 'predict_result': predict_result})


# 显示以前分析结果
def view_diagnose_single(request):
    # 没有登录，返回主页
    if request.session.get('username', None) is None:
        file_view = 'homepage.html'
        return render(request, file_view, {})

    #web发布目录
    baseDir = os.path.dirname(os.path.abspath(__name__))

    #region load result from pkl file.
    str_uuid = request.GET.get('uuid')
    pil_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'rb')
    predict_result = pickle.load(pkl_file)
    #endregion

    #返回分析结果
    lang = str(request.session['lang'])
    if lang == 'en':
        if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
            file_view = 'diagnosis_mc.html'
        else:
            file_view = 'diagnosis_ml.html'
    else:
        if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
            file_view = 'diagnosis_mc_cn.html'
        else:
            file_view = 'diagnosis_ml_cn.html'

    return render(request, file_view,  {'realtime_diagnose': False, 'predict_result': predict_result})


