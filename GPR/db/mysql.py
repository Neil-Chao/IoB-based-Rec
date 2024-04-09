import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pymysql
from pymysql.err import IntegrityError

class MySQL(object):

    connection = None
    cursor = None

    def __init__(self):
        import platform
        system = platform.platform().lower()
        if "windows" in system:
            host="localhost"
            port=3306
        else:
            host="192.168.1.118"
            port=4444
            
        if MySQL.connection is None:
            try:
                MySQL.connection = pymysql.connect(host=host,
                                                        port=port,
                                                        user='root',
                                                        password='root',
                                                        db='oss',
                                                        charset='utf8mb4'
                                                        )
                MySQL.cursor = MySQL.connection.cursor()
            except Exception as error:
                print("Error: Connection not established {}".format(error))
            else:
                print("Connection established")

        self.connection = MySQL.connection
        self.cursor = MySQL.cursor

    def get_group_issue(self, group):
        sql = "SELECT * FROM `issue` WHERE `author` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()

    def get_group_comment(self, group):
        sql = "SELECT * FROM `comment` WHERE `author` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_close(self, group):
        sql = "SELECT * FROM `close` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_commit(self, group):
        sql = "SELECT * FROM `commit` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_link(self, group):
        sql = "SELECT * FROM `issue_pr` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_label(self, group):
        sql = "SELECT * FROM `label` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_mention(self, group):
        sql = "SELECT * FROM `mention` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_merge(self, group):
        sql = "SELECT * FROM `merge` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_quote(self, group):
        sql = "SELECT * FROM `quote` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def get_group_review(self, group):
        sql = "SELECT * FROM `review` WHERE `user` IN %(group)s"
        self.cursor.execute(sql, {'group': group})
        return self.cursor.fetchall()
    
    def insert_single_user(self, username):
        sql = "INSERT INTO `user` VALUES (NULL, %s)"
        try:
            self.cursor.execute(sql, username)
        except IntegrityError:
            pass
    
    def get_user_from_close(self):
        sql = "SELECT DISTINCT `user` FROM `close`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_comment(self):
        sql = "SELECT DISTINCT `author` FROM `comment`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_commit(self):
        sql = "SELECT DISTINCT `user` FROM `commit`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_issue(self):
        sql = "SELECT DISTINCT `author` FROM `issue`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_issue_pr(self):
        sql = "SELECT DISTINCT `user` FROM `issue_pr`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_label(self):
        sql = "SELECT DISTINCT `user` FROM `label`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_merge(self):
        sql = "SELECT DISTINCT `user` FROM `merge`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_pr(self):
        sql = "SELECT DISTINCT `author` FROM `pr`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_review(self):
        sql = "SELECT DISTINCT `user` FROM `review`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_user_from_mention(self):
        sql = "SELECT DISTINCT `user` FROM mention WHERE source LIKE 'https://github.com/facebook/react/%'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_users(self):
        sql = "SELECT `username` FROM `user`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_issues(self):
        sql = "SELECT * FROM `issue`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_prs(self):
        sql = "SELECT * FROM `pr`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_close_for_url(self, url):
        sql = "SELECT `user`, `operate_time`, `url`, `close`, `lock` FROM `close` WHERE `url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_close_for_username(self, username):
        sql = "SELECT `user`, `operate_time`, `url`, `close`, `lock` FROM `close` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_comment_for_url(self, url):
        sql = "SELECT `author`, `datetime`, `url` FROM `comment` WHERE `url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_comment_for_username(self, username):
        sql = "SELECT `author`, `datetime`, `url` FROM `comment` WHERE `author`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_comment_from_now(self, username, datetime):
        sql = "SELECT `author`, ABS(TIMESTAMPDIFF(SECOND, `datetime`, %s)) as diff, `url` FROM `comment` WHERE `author`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_issue_for_url(self, url):
        sql = "SELECT `author`, `open_time`, `url` FROM `issue` WHERE `url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_issue_for_url(self, username):
        sql = "SELECT `author`, `open_time`, `url` FROM `issue` WHERE `author`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_issue(self, url):
        sql = "SELECT `user`, `link_time`, `issue_url`, `is_link` FROM `issue_pr` WHERE `issue_url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_pr(self, url):
        sql = "SELECT `user`, `link_time`, `pr_url`, `is_link` FROM `issue_pr` WHERE `pr_url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_username(self, username):
        sql = "SELECT `user`, `link_time`, `issue_url`, `is_link` FROM `issue_pr` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_username_from_issue(self, username):
        sql = "SELECT `user`, `link_time`, `issue_url`, `is_link` FROM `issue_pr` WHERE `issue_url` LIKE 'https://github.com/facebook/react/issues/%%' AND `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_username_from_issue_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `link_time`, %s)) as diff, `issue_url`, `is_link` FROM `issue_pr` WHERE `issue_url` LIKE 'https://github.com/facebook/react/issues/%%' AND `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_username_from_pr(self, username):
        sql = "SELECT `user`, `link_time`, `pr_url`, `is_link` FROM `issue_pr` WHERE `user`=%s AND `pr_url` LIKE 'https://github.com/facebook/react/pull/%%'"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_issue_pr_for_username_from_pr_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `link_time`, %s)) as diff, `pr_url`, `is_link` FROM `issue_pr` WHERE `pr_url` LIKE 'https://github.com/facebook/react/pull/%%' AND `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_label_for_url(self, url):
        sql = "SELECT `user`, `label_time`, `url`, `delete` FROM `label` WHERE `url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_label_for_username(self, username):
        sql = "SELECT `user`, `label_time`, `url`, `delete` FROM `label` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_label_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `label_time`, %s)) as diff, `url`, `delete` FROM `label` WHERE `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_mention_for_url(self, url):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `source`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_mention_for_username(self, username):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_mention_for_username_from_issue(self, username):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `user`=%s AND `source` LIKE 'https://github.com/facebook/react/issues/%%'"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_mention_for_username_from_issue_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `mention_time`, %s)) as diff, `source` FROM `mention` WHERE `source` LIKE 'https://github.com/facebook/react/issues/%%' AND `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_mention_for_username_from_pr(self, username):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `user`=%s AND `source` LIKE 'https://github.com/facebook/react/pull/%%'"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_mention_for_username_from_pr_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `mention_time`, %s)) as diff, `source` FROM `mention` WHERE `source` LIKE 'https://github.com/facebook/react/pull/%%' AND `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_commit_for_url(self, url):
        sql = "SELECT `user`, `commit_time`, `pr_url` FROM `commit` WHERE `pr_url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_commit_for_username(self, username):
        sql = "SELECT `user`, `commit_time`, `pr_url` FROM `commit` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_commit_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `commit_time`, %s)) as diff, `pr_url` FROM `commit` WHERE `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_merge_for_url(self, url):
        sql = "SELECT `user`, `merge_time`, `pr_url` FROM `merge` WHERE `pr_url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_merge_for_username(self, username):
        sql = "SELECT `user`, `merge_time`, `pr_url` FROM `merge` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_review_for_url(self, url):
        sql = "SELECT `user`, `review_time`, `pr_url`, `dismiss` FROM `review` WHERE `pr_url`=%s"
        self.cursor.execute(sql, (url))
        return self.cursor.fetchall()
    
    def get_review_for_username(self, username):
        sql = "SELECT `user`, `review_time`, `pr_url`, `dismiss` FROM `review` WHERE `user`=%s"
        self.cursor.execute(sql, (username))
        return self.cursor.fetchall()
    
    def get_review_from_now(self, username, datetime):
        sql = "SELECT `user`, ABS(TIMESTAMPDIFF(SECOND, `review_time`, %s)) as diff, `pr_url`, `dismiss` FROM `review` WHERE `user`=%s ORDER BY diff LIMIT 5"
        self.cursor.execute(sql, (datetime, username))
        return self.cursor.fetchall()
    
    def get_user_id(self, username):
        sql = "SELECT `id` FROM `user` WHERE `username`=%s"
        try:
            self.cursor.execute(sql, (username))
        except:
            print(1)
        return self.cursor.fetchall()
    
    def get_username(self, uid):
        sql = "SELECT `username` FROM `user` WHERE `id`=%s"
        try:
            self.cursor.execute(sql, (uid))
        except:
            print(1)
        return self.cursor.fetchall()
    
    def get_all_comments(self):
        sql = "SELECT `author`, `datetime`, `url` FROM `comment`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_commits(self):
        sql = "SELECT `user`, `commit_time`, `pr_url` FROM `commit`"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_review(self):
        sql = "SELECT `user`, `review_time`, `pr_url` FROM `review` WHERE `dismiss`=0"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_label(self):
        sql = "SELECT `user`, `label_time`, `url` FROM `label` WHERE `delete`=0"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_mentions_from_issue(self):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `source` LIKE 'https://github.com/facebook/react/issues/%'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_mentions_from_pr(self):
        sql = "SELECT `user`, `mention_time`, `source` FROM `mention` WHERE `source` LIKE 'https://github.com/facebook/react/pull/%'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_links_from_issue(self):
        sql = "SELECT `user`, `link_time`, `issue_url` FROM `issue_pr` WHERE `issue_url` LIKE 'https://github.com/facebook/react/issues/%'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_all_links_from_pr(self):
        sql = "SELECT `user`, `link_time`, `pr_url` FROM `issue_pr` WHERE `pr_url` LIKE 'https://github.com/facebook/react/pull/%'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    

if __name__ == "__main__":
    mysql = MySQL()
    from datetime import datetime
    # print(mysql.get_comment_from_now('gaearon', datetime.now()))
    sql = "SELECT `author`, ABS(TIMESTAMPDIFF(SECOND, `datetime`, %s)) as diff, `url` FROM `comment` WHERE `author`=%s ORDER BY diff LIMIT 5"
    mysql.cursor.execute(sql, (datetime.now(), "gaearon"))
    print(mysql.cursor.fetchall())