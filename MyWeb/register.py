
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response
from django.views.decorators import csrf
import MySQLdb


def show_register(request):
    lang = request.GET.get('lang', 'en')

    if lang == 'en':
        file_view = 'register.html'
    else:
        file_view = 'register_cn.html'

    return render(request, file_view, {})


def do_register(request):
    lang = request.POST.get('lang', 'en')

    email = request.POST.get('email')
    name = request.POST.get('name')
    tel = request.POST.get('tel')
    company = request.POST.get('company')
    title = request.POST.get('title')

    if not '@' in email or email == '' or name == '' or company == '' or title == '':
        if lang == 'en':
            return render(request, 'register.html', {'error_msg':'Please fill in the form correctly!'})
        else:
            return render(request, 'register_cn.html', {'error_msg':'请按照规范填写表单'})

    from my_module import db_helper
    db = db_helper.get_db_conn()

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 参数化处理
    sql_check_email = "SELECT * FROM tb_account WHERE username=%s and enabled=1"
    cursor.execute(sql_check_email, (email,))
    results = cursor.fetchall()

    if len(results) > 0:    #邮件账号已经使用
        db.close()

        if lang == 'en':
            return render(request, 'register.html', {'error_msg': 'The mail has been used!!'})
        else:
            return render(request, 'register_cn.html', {'error_msg': '该邮件地址已经被使用！'})

    else:
        sql_insert = 'insert into tb_register(email, password, name, tel, company, title) values(%s, %s, %s, %s, %s, %s)'

        cursor.execute(sql_insert, (email, '', name,  tel, company, title))
        db.commit()
        db.close()

        if lang == 'en':
            return render(request, 'register_ok.html', {})
        else:
            return render(request, 'register_ok_cn.html', {})