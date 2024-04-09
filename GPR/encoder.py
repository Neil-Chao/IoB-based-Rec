import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from GPR.db.mysql import MySQL
import GPR.config.constant as CONSTANT

import json
import functools
from datetime import datetime

class Encoder():
    '''
    Behavior Type:
        - issue 0
        - description 1
        - ask 2
        - answer 3
        - suggest 4
        - agree 5
        - reply 6
        - disagree 7
        - likewise 8
        - other 9
        - close 10
        - reopen 11
        - lock 12
        - pr 13
        - commit 14
        - link 15
        - unlink 16
        - label 17
        - unlabel 18
        - mention 19
        - merge 20
        - quote 21
        - review 22
        - dismiss 23
    '''

    def __init__(self, group) -> None:
        self.group: list = group
        self.size = len(group)
        self.mysql = MySQL()
    
    '''
    0) id
    1) datetime
    2) user_type
    3) author
    4) heart
    5) rocket
    6) down
    7) up
    8) laugh
    9) hooray
    10) confused
    11) eyes
    12) off_topic
    13) content
    14) comment_type
    15) url
    16) at
    17) quotes
    '''
    def extract_comment(self):
        comments = self.mysql.get_group_comment(self.group)
        b_list = []
        for comment in comments:
            obj = []
            # type
            obj.append(json.loads(comment[14]))
            # author
            obj.append(comment[3])
            # datetime
            obj.append(comment[1])
            # repository
            obj.append(comment[15])
            # role
            obj.append(json.loads(comment[2]))
            
            # content
            obj.append(comment[13])

            # like
            obj.append(comment[4])
            obj.append(comment[5])
            obj.append(comment[6])
            obj.append(comment[7])
            obj.append(comment[8])
            obj.append(comment[9])
            obj.append(comment[10])
            obj.append(comment[11])

            b_list.append(obj)
            
        return b_list


    '''
    0) id
    1) operate_time
    2) user
    3) url
    4) close
    5) lock
    '''
    def extract_close(self):
        closes = self.mysql.get_group_close(self.group)
        b_list = []
        for close in closes:
            obj = []

            # type
            if close[5] == 1:
                if close[6] == 1:
                    obj.append([12])
                else:
                    obj.append([10])
            else:
                obj.append([11])

            # user
            obj.append(close[2])
            # datetime
            obj.append(close[1])
            # repository
            obj.append(close[3])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list


    '''
    0) id
    1) user
    2) comment
    3) commit_time
    4) pr_url
    '''
    def extract_commit(self):
        commits = self.mysql.get_group_commit(self.group)
        b_list = []
        for commit in commits:
            obj = []

            # type
            obj.append([14])

            # user
            obj.append(commit[1])
            # datetime
            obj.append(commit[3])
            # repository
            obj.append(commit[4])
            # role
            obj.append([])
            
            # content
            obj.append(commit[2])
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list


    '''
    0) id
    1) issue_url
    2) pr_url
    3) link_time
    4) user
    5) is_link
    '''
    def extract_link(self):
        links = self.mysql.get_group_link(self.group)
        b_list = []
        for link in links:
            
            obj = []

            # type
            if link[5] == 1:
                obj.append([15])
            else:
                obj.append([16])

            # user
            obj.append(link[4])
            # datetime
            obj.append(link[3])
            # repository
            obj.append(link[2])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list
    
    '''
    0) id
    1) label
    2) url
    3) user
    4) label_time
    5) delete
    '''
    def extract_label(self):
        labels = self.mysql.get_group_label(self.group)
        b_list = []
        for label in labels:
            obj = []

            # type
            if label[5] == 1:
                obj.append([18])
            else:
                obj.append([17])

            # user
            obj.append(label[3])
            # datetime
            obj.append(label[4])
            # repository
            obj.append(label[2])
            # role
            obj.append([])
            
            # content
            obj.append(label[1])
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list
    

    '''
    0) id
    1) source
    2) target
    3) user
    4) mention_time
    '''
    def extract_mention(self):
        mentions = self.mysql.get_group_mention(self.group)
        b_list = []
        for mention in mentions:
            obj = []

            # type
            obj.append([19])

            # user
            obj.append(mention[3])
            # datetime
            obj.append(mention[4])
            # repository
            obj.append(mention[1])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list


    '''
    0) id
    1) user
    2) pr_url
    3) merge_time
    '''
    def extract_merge(self):
        merges = self.mysql.get_group_merge(self.group)
        b_list = []
        for merge in merges:
            obj = []

            # type
            obj.append([20])

            # user
            obj.append(merge[1])
            # datetime
            obj.append(merge[3])
            # repository
            obj.append(merge[2])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list


    '''
    未定
    '''
    def extract_quote(self):
        merges = self.mysql.get_group_quote(self.group)
        b_list = []
        for merge in merges:
            obj = []

            # type
            obj.append([20])

            # user
            obj.append(merge[1])
            # datetime
            obj.append(merge[3])
            # repository
            obj.append(merge[2])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list


    '''
    0) id
    1) user
    2) pr_url
    3) merge_time
    '''
    def extract_merge(self):
        merges = self.mysql.get_group_merge(self.group)
        b_list = []
        for merge in merges:
            obj = []

            # type
            obj.append([20])

            # user
            obj.append(merge[1])
            # datetime
            obj.append(merge[3])
            # repository
            obj.append(merge[2])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list
    

    '''
    0) id
    1) review_time
    2) user
    3) pr_url
    4) dismiss
    '''
    def extract_review(self):
        merges = self.mysql.get_group_review(self.group)
        b_list = []
        for merge in merges:
            obj = []

            # type
            if merge[4] == 1:
                obj.append([23])
            else:
                obj.append([22])

            # user
            obj.append(merge[2])
            # datetime
            obj.append(merge[1])
            # repository
            obj.append(merge[3])
            # role
            obj.append([])
            
            # content
            obj.append(None)
            # like
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)
            obj.append(0)

            b_list.append(obj)
            
        return b_list

    '''
    Behavior attribute
        - Behavior type
        - User
        - Datetime
        - Repository
        - Role
        - Content
        - Like
    '''
    def extract_behavior(self):
        res =     self.extract_close() \
                + self.extract_comment() \
                + self.extract_commit() \
                + self.extract_label() \
                + self.extract_link() \
                + self.extract_mention() \
                + self.extract_merge() \
                + self.extract_quote() \
                + self.extract_review()
        
        res.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[2].__ge__(b[2]) else -1))

        return res
        
    def encode(self, b_sequence):
        res = []
        for b in b_sequence:
            res.append(self._encode_b(b))
        return res
        
    def _encode_b(self, b):
        one_hot = []
        one_hot += self.__handle_type(b[0])
        one_hot += self.__handle_user(b[1])
        one_hot += self.__handle_datetime(b[2])

        one_hot += self.__handle_role(b[4])
        one_hot += b[-8:]
        return one_hot
    
    def __handle_type(self, type_list):
        emb = [0 for _ in range(CONSTANT.TOTAL_TYPE)]
        for t in type_list:
            emb[t] = 1
        return emb
    
    def __handle_user(self, user):
        emb = [0 for _ in range(self.size)]
        emb[self.group.index(user)] = 1
        return emb
    
    def __handle_datetime(self, datetime: datetime):
        return [datetime.timestamp()]
    
    def __handle_role(self, type_list):
        emb = [0 for _ in range(CONSTANT.TOTAL_ROLE)]
        for t in type_list:
            emb[t] = 1
        return emb
    

if __name__ == "__main__":
    group = ["sophiebits", "eps1lon", "yairopro", "denmo530", "hoxyq"]
    encoder = Encoder(group)
    print(encoder.encode(encoder.extract_behavior()))
        