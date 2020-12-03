'''
    API for web
    API for PACS client
'''

import os
from django.shortcuts import render
import uuid
from django.http import HttpResponse
import pickle
import cv2

#AI分析的类, 数据库连接类
import my_module.my_dlp_helper_multi_class
import my_module.my_dlp_helper_multi_labels
from my_module import my_dlp_helper, db_helper
import shutil
import my_config

#API test web page
def uploadfile(request):
    file_view = 'api_uploadfile.html'
    return render(request, file_view, {})

# API Service, 浏览器上传图像文件，返回uuid，然后调用分析服务
def uploadfile_service(request):
    import json
    try:
        # region login
        username = request.POST.get('username')
        password = request.POST.get('password')

        login_OK = db_helper.login(username, password, write_log=True,
                                   source_ip=request.META['REMOTE_ADDR'])

        if login_OK:
            request.session['username'] = username
        else:
            data = {'err_code': 1, 'err_message': "username or passrod error"}
            json = json.dumps(data)
            return HttpResponse(json)
        #endregion

        #web发布目录
        baseDir = os.path.dirname(os.path.abspath(__name__))

        if request.FILES.get('input_image_file') == None:
            data = {'err_code': 2, 'err_message': "no image file"}
            json = json.dumps(data)
            return HttpResponse(json)
        else:
            # 生成一个本次分析的ID
            str_uuid = str(uuid.uuid1())

            # 接收文件
            request_fime = request.FILES.get('input_image_file')

            # 保存文件的目录
            jpgdir = os.path.join(baseDir, 'static', 'imgs', str_uuid)
            os.makedirs(jpgdir, exist_ok=True)
            filename = os.path.join(jpgdir, request_fime.name)

            fobj = open(filename, 'wb')
            for chrunk in request_fime.chunks():
                fobj.write(chrunk)
            fobj.close()

            filename_orig_new = os.path.join(jpgdir, 'original.jpg')
            img_temp = cv2.imread(filename)
            cv2.imwrite(filename_orig_new, img_temp)

            data = {'err_code': 0, 'str_uuid': str_uuid}
            json = json.dumps(data)

            return HttpResponse(json)
    except Exception as ex:
        print(ex)
        data = {'err_code': 3, 'err_message': "other error"}
        json = json.dumps(data)
        return HttpResponse(json)


# API Service, 已经上传了文件，调用分析接口
# api_diagnose_service?username=test&password=jsiec&showcam=1&uuid=ddd
def diagnose_service(request):
    import json

    #region login
    username = request.GET.get('username')
    password = request.GET.get('password')

    login_OK = db_helper.login(username, password, write_log=True,
                               source_ip=request.META['REMOTE_ADDR'])

    if not login_OK:
        data = {'err_code': 1, 'err_message': "username or passrod error"}
        json = json.dumps(data)
        return HttpResponse(json)

    if 'HTTP_X_FORWARDED_FOR' in request.META:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.META['REMOTE_ADDR']

    #endregion

    lang = request.GET.get('lang', 'cn')

    #web发布目录
    baseDir = os.path.dirname(os.path.abspath(__name__))

    cam_type = request.GET.get('cam_type', '1')

    str_uuid = request.GET.get('str_uuid')
    if str_uuid is None:
        data = {'err_code': 2, 'err_message': "no str_uuid parameter"}
        json = json.dumps(data)
        return HttpResponse(json)

    jpgdir = os.path.join(baseDir, 'static', 'imgs', str_uuid)
    filename_full_path = os.path.join(jpgdir, 'original.jpg')

    if not os.path.exists(filename_full_path):
        data = {'err_code': 3, 'err_message': "image file not found!"}
        json = json.dumps(data)
        return HttpResponse(json)

    # predict_result = my_dlp_helper.predict_all_multi_labels(str_uuid, filename_full_path, baseDir, lang, show_cam=showcam)
    deeplift = request.GET.get('deeplift', '0')
    deepshap = request.GET.get('deepshap', '0')
    if deepshap == '0':
        show_deeplift = False
    else:
        show_deeplift = True
    if deeplift == '0':
        show_deepshap = False
    else:
        show_deepshap = True


    if my_config.SOFTMAX_OR_SIGMOIDS == 'softmax':
        predict_result = my_module.my_dlp_helper_multi_class.predict_all_multi_class(str_uuid, filename_full_path, baseDir, lang,
                                                                                     cam_type=cam_type, show_deeplift=show_deeplift, show_deepshap=show_deepshap)
    else:
        predict_result = my_module.my_dlp_helper_multi_labels.predict_all_multi_labels(str_uuid, filename_full_path, baseDir, lang,
                                                                                       cam_type=cam_type, show_deeplift=show_deeplift, show_deepshap=show_deepshap)

    #region 保存结果
    # 保存分析结果，以后直接调出
    pil_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'wb')
    pickle.dump(predict_result, pkl_file)

    #写入数据库
    diagnostic_results = predict_result['disease_name']

    # 一个uuid重复分析
    my_dlp_helper.save_to_db_diagnose(ip, username, str_uuid, diagnostic_results, del_duplicate=True)
    #endregion

    #返回分析结果
    json = json.dumps(predict_result)
    return HttpResponse(json)

'''

def diagnose_service(request):
    import json

    #region login
    username = request.GET.get('username')
    password = request.GET.get('password')

    login_OK = db_helper.login(username, password, write_log=True,
                               source_ip=request.META['REMOTE_ADDR'])

    if login_OK:
        request.session['username'] = username
    else:
        data = {'err_code': 1, 'err_message': "username or passrod error"}
        json = json.dumps(data)
        return HttpResponse(json)

    if 'HTTP_X_FORWARDED_FOR' in request.META:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.META['REMOTE_ADDR']

    import my_config
    request.session['softmax_or_sigmoids'] = my_config.SOFTMAX_OR_SIGMOIDS

    #endregion

    #检测语言版本
    lang = request.GET.get('lang', 'cn')
    request.session['lang'] = lang

    #web发布目录
    baseDir = os.path.dirname(os.path.abspath(__name__))

    if request.GET.get('showcam', '1') == '1':
        showcam = True
    else:
        showcam = False

    str_uuid = request.GET.get('uuid', 'nonono')
    jpgdir = os.path.join(baseDir, 'static', 'imgs', str_uuid)
    filename_full_path = os.path.join(jpgdir, 'original.jpg')

    if not os.path.exists(filename_full_path):
        return HttpResponse('Error!')

    request.session['image_uuid'] = str_uuid

    # predict_result = my_dlp_helper.predict_all_multi_labels(str_uuid, filename_full_path, baseDir, lang, show_cam=showcam)

    if request.session.get('softmax_or_sigmoids', 'softmax') == 'softmax':
        predict_result = my_dlp_helper.predict_all_multi_class(str_uuid, filename_full_path, baseDir, lang,
                                                               show_cam=showcam)
    else:
        predict_result = my_dlp_helper.predict_all_multi_labels(str_uuid, filename_full_path, baseDir, lang,
                                                               show_cam=showcam)

    #region 保存结果
    # 保存分析结果，以后直接调出
    pil_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'wb')
    pickle.dump(predict_result, pkl_file)

    #写入数据库
    diagnostic_results = predict_result['disease_name']

    # 一个uuid重复分析
    my_dlp_helper.save_to_db_diagnose(ip, username, str_uuid, diagnostic_results, del_duplicate=True)
    #endregion

    #返回分析结果

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

'''

# PACS Service   pacs_service?checkno=1981577
def pacs_service(request):
    if request.method == 'GET' and ('checkno' in request.GET):

        checkno = request.GET['checkno']
        lang = request.GET.get('lang', 'cn')

        img_base_dir = os.path.join(my_config.BASE_DIR_PACS_SERVICE_INPUT, checkno)
        if not os.path.exists(img_base_dir):
            return HttpResponse('This checkno does not exist!')

        # echo 1 | sudo -S chmod  -R  777 1993381
        import subprocess
        subprocess.call('echo 1 | sudo -S chmod  -R  777 ' + img_base_dir, shell=True)
        # import stat
        # os.chmod(img_base_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)  # mode:777

        #delete old dir if exists
        dir_save = os.path.join(my_config.BASE_DIR_PACS_SERVICE_OUTPUT, checkno)
        if os.path.exists(dir_save):
            shutil.rmtree(dir_save)  # os.removedirs(dir_save) do not work

        for f in os.scandir(img_base_dir):
            if os.path.isdir(f.path):  # every file's results save to a subdir(filename as dirname)
                continue

            file_original = os.path.join(img_base_dir, f)
            my_dlp_helper.predict_all_pacs(checkno=checkno, file_img_source=file_original,
                                           baseDir=my_config.BASE_DIR_PACS_SERVICE_OUTPUT,
                                           lang=lang,
                                           cam_type='1', show_deeplift=False, show_deepshap=False)

        return HttpResponse('OK!')
    else:
        return HttpResponse('Request Error!')


#和相机集成，相机微信小程序先ftp单个文件，然后调用服务
# CAMERA ftp 文件存放位置
BASE_DIR_CAMERA_SERVICE_UPLOAD = '/home/ftp_fundus_image/upload/'
BASE_DIR_CAMERA_SERVICE_RESULTS = '/home/ftp_fundus_image/results/'

def camera_service(request):
    if request.method == 'GET' and ('img_id' in request.GET):
        img_id = request.GET['img_id']

        get_CAM = True  #默认 生成热力图
        if 'getCAM' in request.GET:
            if (request.GET['getCAM'] == '0') or (request.GET['getCAM'] == ''):
                get_CAM = False

        # 语言版本
        lang = request.GET.get('lang', 'cn')

        file_original = os.path.join(BASE_DIR_CAMERA_SERVICE_UPLOAD, img_id+'.jpg')

        if not os.path.exists(file_original):
            return HttpResponse('This file does not exist!')

        #更改目录权限 sudo chmod -R 777 ./PACS/original/
        # echo ecadmin | sudo -S chmod  -R  777 1993381
        import subprocess
        subprocess.call('echo ecadmin | sudo -S chmod  -R  777 ' + file_original, shell=True)
        # echo ecadmin | sudo -S chmod  -R  777 /home/jsiec/disk1/PACS/original/1991060
        # import stat
        # os.chmod(img_base_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)  # mode:777

        dir_results = os.path.join(BASE_DIR_CAMERA_SERVICE_RESULTS, img_id)

        #分析 (只有baseDir,showcam,is_web_client,file_base_name参数有用，其他参数是给web client的)
        my_module.my_dlp_helper_multi_labels.predict_all_multi_labels(file_img_source=file_original, str_uuid=img_id, baseDir=dir_results, lang=lang,
                                                                      showcam=get_CAM, is_web_client=False)

        return HttpResponse('OK!')
    else:
        return HttpResponse('Request Error!')

