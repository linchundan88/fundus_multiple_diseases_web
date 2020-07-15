# -*- coding: utf-8 -*-
'''显示分析结果后，用户反馈意见
'''
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response
from django.views.decorators import csrf
import MySQLdb

def op_feedback(request):
    if (request.session.get('image_uuid', None) is None)\
            or (request.session.get('username', None) is None):
        file_view = 'error.html'
        return render(request, file_view, {})

    image_uuid = request.session.get('image_uuid')

    #region 写入数据库
    feedback_score = request.GET.get('score')
    feedback_memo = request.GET.get('memo')

    db = MySQLdb.connect("localhost", "dlp", "dlp13502965818", "AI", use_unicode=True, charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 参数化处理
    sql = "update tb_diagnoses set feedback_score=%s, feedback_memo=%s  where image_uuid=%s "
    cursor.execute(sql, (feedback_score, feedback_memo, image_uuid))
    db.commit()
    db.close()
    #endregion

    # ender是渲染变量到模板中,而redirect是HTTP中的1个跳转的函数,一般会生成302状态码
    # return HttpResponseRedirect('/view_diagnoses')

    lang = str(request.session['lang'])
    if lang == 'en':
        # 英文版
        file_view = 'feedback_ok.html'
    else:
        # 中文版
        file_view = 'feedback_ok_cn.html'

    return render(request, file_view, {})
