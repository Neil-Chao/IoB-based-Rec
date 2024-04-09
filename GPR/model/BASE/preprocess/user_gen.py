import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from GPR.db.mysql import MySQL


def gen_from_close():
    mysql = MySQL()
    res = mysql.get_user_from_close()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_comment():
    mysql = MySQL()
    res = mysql.get_user_from_comment()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

    
def gen_from_commit():
    mysql = MySQL()
    res = mysql.get_user_from_commit()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_issue():
    mysql = MySQL()
    res = mysql.get_user_from_issue()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_issue_pr():
    mysql = MySQL()
    res = mysql.get_user_from_issue()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_label():
    mysql = MySQL()
    res = mysql.get_user_from_label()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_merge():
    mysql = MySQL()
    res = mysql.get_user_from_merge()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_pr():
    mysql = MySQL()
    res = mysql.get_user_from_pr()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_review():
    mysql = MySQL()
    res = mysql.get_user_from_review()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()

def gen_from_mention():
    mysql = MySQL()
    res = mysql.get_user_from_mention()
    for username in res:
        mysql.insert_single_user(username)
    mysql.connection.commit()


if __name__ == "__main__":
    # gen_from_close()
    # gen_from_comment()
    # gen_from_commit()
    # gen_from_issue()
    gen_from_issue_pr()
    # gen_from_label()
    # gen_from_merge()
    # gen_from_pr()
    # gen_from_review()
    # gen_from_mention()