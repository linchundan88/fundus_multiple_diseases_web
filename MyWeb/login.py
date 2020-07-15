
'''用户登录页面
login登录
logout 注销
'''

from django.shortcuts import render
from django.http import HttpResponseRedirect
from my_module import db_helper

def homePage(request):
    file_view = 'homepage.html'
    return render(request, file_view, {})

def login(request):
    #already login
    if request.session.get('username', None) is not None:
        return HttpResponseRedirect('/uploadimage')

    # homepage.html <a href="/login?lang=en">
    if request.method == 'GET':
        lang = request.GET.get('lang', 'en')

        if lang == 'en':
            file_view = 'login.html'
        else:
            file_view = 'login_cn.html'
        return render(request, file_view, {})
    else:
        # login validation
        lang = request.POST.get('lang', 'en')

        username = request.POST.get('username')
        password = request.POST.get('password')

        if 'HTTP_X_FORWARDED_FOR' in request.META:
            ip = request.META['HTTP_X_FORWARDED_FOR']
        else:
            ip = request.META['REMOTE_ADDR']

        login_OK = db_helper.login(username, password, write_log=True,
                                   source_ip=ip)
        if login_OK:
            request.session['username'] = username
            request.session['lang'] = lang

            import my_config
            request.session['softmax_or_sigmoids'] = my_config.SOFTMAX_OR_SIGMOIDS

            if lang == 'en':
                file_view = 'uploadfile.html'
            else:
                file_view = 'uploadfile_cn.html'
            return render(request, file_view, {})
        else: # login failure
            if lang == 'en':
                return render(request, 'login.html',
                              {'err_msg': 'username or password error!'})
            else:
                return render(request, 'login_cn.html',
                              {'err_msg': '账号或者密码错误!'})

def register(request):
    lang = request.POST.get('lang', 'en')

    if lang == 'en':
        file_view = 'register.html'
    else:
        file_view = 'register_cn.html'

    return render(request, file_view, {})

def show_upload(request):

    # do not login
    if request.session.get('username', None) is None:
        return render(request, 'homepage.html')

    # 显示上传图像页面
    lang =request.session.get('lang')
    if lang == 'en':
        # 英文版
        file_view = 'uploadfile.html'
    else:
        # 中文版
        file_view = 'uploadfile_cn.html'
    return render(request, file_view, {})

def logout(request):
    del request.session['username']
    del request.session['lang']

    return render(request, 'homepage.html', {})
