'''
#python3 MySQLdb
sudo apt-get install python-dev libmysqlclient-dev
sudo apt-get install python3-dev
pip install mysqlclient
'''

DB_TYPE = 'sqlite'  # mysql

import MySQLdb
def get_db_conn_mysql():
    # conn = MySQLdb.connect("10.12.192.135", "dlp", "dlp13502965818", "AI", use_unicode=True, charset='utf8')
    conn = MySQLdb.connect('localhost', "dlp", "dlp13502965818", "AI", use_unicode=True, charset='utf8')
    return conn

import os
import sys
import sqlite3
def get_db_conn():
    db_file = os.path.join(sys.path[0], 'database', 'dlp.sqlite')
    conn = sqlite3.connect(db_file)
    return conn

def login(username, password, write_log=True, source_ip='127.0.0.1'):
    db = get_db_conn()
    cursor = db.cursor()

    from my_module.my_compute_digest import CalcSha1_str
    password_encrypt = CalcSha1_str(password)

    if DB_TYPE == 'mysql':
        sql = "SELECT * FROM tb_account WHERE username=%s and password_encrypt=%s and enabled=1"
    if DB_TYPE == 'sqlite':
        sql = "SELECT * FROM tb_account WHERE username=? and password_encrypt=? and enabled=1"
    cursor.execute(sql, (username, password_encrypt))
    results = cursor.fetchall()

    if len(results) == 1:
        if write_log:
            if DB_TYPE == 'mysql':
                sql = "insert into tb_log(username,log_memo) values(%s,%s)"
            if DB_TYPE == 'sqlite':
                sql = "insert into tb_log(username,log_memo) values(?,?)"
            cursor.execute(sql, (username, 'login_successful, ip:' + source_ip))
            db.commit()

        return True
    else:
        if write_log:
            if DB_TYPE == 'mysql':
                sql = "insert into tb_log(username,log_memo) values(%s,%s)"
            if DB_TYPE == 'sqlite':
                sql = "insert into tb_log(username,log_memo) values(?,?)"
            cursor.execute(sql, (username, 'login_failure, ip:' + source_ip))
            db.commit()

        return False

    db.close()
