'''
uploadfile 上传文件的页面
diagnose 接受上传的文件，进行分析，分析结果调用模版diagnosis.html
用户点击返回 /uploadimage
或者填写意见 提交到/feedback
feedback调用模版feedback_ok.html显示提交成功，然后返回/uploadimage
view_diagnoses 是分析结果列表（历史记录）, 调用模版view_diagnoses

pacs_service是给pccs的接口
'''

from django.conf.urls import url
from django.contrib import admin
from . import ai
from . import ai_api
from . import feedback
from . import login
from . import register
from . import view_diagnoses
from . import saliency

urlpatterns = [
    url(r'^login', login.login),
    url(r'^register', register.show_register),
    url(r'^do_register', register.do_register),
    url(r'^logout', login.logout),

    url(r'^api_uploadfile_test', ai_api.uploadfile),
    # API Service, firsty client upload an image to the system(return uuid）
    url(r'^api_uploadfile_service', ai_api.uploadfile_service),
    # and then call diagnose_service using uuid
    url(r'^api_diagnose_service', ai_api.diagnose_service),

    url(r'^uploadimage', login.show_upload),
    url(r'^diagnose', ai.diagnose),

    # ajax service for CAM and deeplift
    url(r'^saliency', saliency.get_saliency_map),

    url(r'^feedback', feedback.op_feedback),
    url(r'^view_diagnoses_single', ai.view_diagnose_single),
    url(r'^view_diagnoses', view_diagnoses.view_diagnoses_list),

    url(r'^pacs_service', ai_api.pacs_service),

    url(r'^camera_service', ai_api.camera_service),

    url(r'^$', login.homePage),

]

