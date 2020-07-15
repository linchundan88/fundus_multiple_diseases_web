import os
import pickle
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
import MySQLdb
import pickle

def view_diagnoses_list(request):
    # 没有登录，返回主页
    if request.session.get('username', None) is None:
        file_view = 'homepage.html'
        return render(request, file_view, {})

    username = request.session.get('username')

    from my_module import db_helper
    db = db_helper.get_db_conn()
    cursor = db.cursor()

    if db_helper.DB_TYPE == 'mysql':
        sql = 'select image_uuid,diagnostic_results,feedback_score,feedback_memo, DATE_FORMAT(date_time, "%%Y-%%m-%%d %%k:%%i:%%s") from tb_diagnoses WHERE username=%s order by date_time DESC '
    if db_helper.DB_TYPE == 'sqlite':
        sql = 'select image_uuid,diagnostic_results,feedback_score,feedback_memo, date_time from tb_diagnoses WHERE username=? order by date_time DESC '

    cursor.execute(sql, (username,))
    results = cursor.fetchall()
    db.close()

    result_list = []
    for row in results:
        temp_list = [row[0], row[1], row[2], row[3], row[4]]
        result_list.append(temp_list)

    lang = request.session.get('lang', 'en')
    if lang == 'en':
        # 英文版
        file_view = 'view_diagnoses_list.html'
    else:
        # 英文版
        file_view = 'view_diagnoses_list_cn.html'

    return render(request, file_view,
                  {'result_list': result_list})

