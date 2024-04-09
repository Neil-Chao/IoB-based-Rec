import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))


from GPR.db.mysql import MySQL
import GPR.config.constant as CONSTANT
import pandas as pd
import functools
from copy import deepcopy
import numpy as np
import json

def generate_raw_sequential_dataset():
    def generate_raw_sequential_record_from_issue(issue, mysql: MySQL, f):
        '''
            uid
            time
            url
            type
        '''
        datas = []
        user_dict = {}
        url = issue[1]
        issue_uid = mysql.get_user_id(issue[3])[0][0]
        datas.append([issue_uid, issue[5], url, 0])
        
        closes = mysql.get_close_for_url(url)
        for close in closes:
            uid = mysql.get_user_id(close[0])[0][0]
            if close[4] == 0:
                datas.append([uid, close[1], url, 4])
            elif close[4] == 1:
                datas.append([uid, close[1], url, 5])
            else:
                if close[3] == 1:
                    datas.append([uid, close[1], url, 2])
                else:
                    datas.append([uid, close[1], url, 3])
        
        comments = mysql.get_comment_for_url(url)
        for comment in comments:
            uid = mysql.get_user_id(comment[0])[0][0]
            datas.append([uid, comment[1], url, 1])

        issue_prs = mysql.get_issue_pr_for_issue(url)
        for issue_pr in issue_prs:
            uid = mysql.get_user_id(issue_pr[0])[0][0]
            if issue_pr[3] == 1:
                datas.append([uid, issue_pr[1], url, 8])
            else:
                datas.append([uid, issue_pr[1], url, 9])

        labels = mysql.get_label_for_url(url)
        for label in labels:
            uid = mysql.get_user_id(label[0])[0][0]
            if label[3] == 0:
                datas.append([uid, label[1], url, 10])
            else:
                datas.append([uid, label[1], url, 11])

        mentions = mysql.get_mention_for_url(url)
        for mention in mentions:
            uid = mysql.get_user_id(mention[0])[0][0]
            datas.append([uid, mention[1], url, 12])
        

        if len(datas) < 6:
            return None
        datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))

        for record in datas:
            uid = record[0]
            if uid not in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(record[3])

        res = []
        
        for key, value in user_dict.items():
            if(len(value) < 2):
                continue
            arr = [key] + [x + 1 for x in value]
            arr = [str(x) for x in arr]
            f.write(" ".join(arr))
            f.write("\n")

            # for i in range(1, len(value)):
            #     if(i - 5 < 0):
            #         res.append([key, [0] * (5 - i) + value[0:i], value[i]])
            #     else:
            #         res.append([key, value[i-5:i], value[i]])
        return res
    
    def generate_sequential_record_from_pr(pr, mysql: MySQL, f):
        '''
            uid
            time
            url
            type
        '''
        datas = []
        user_dict = {}
        url = pr[3]
        pr_uid = mysql.get_user_id(pr[2])[0][0]
        datas.append([pr_uid, pr[5], url, 6])

        closes = mysql.get_close_for_url(url)
        for close in closes:
            uid = mysql.get_user_id(close[0])[0][0]
            if close[4] == 0:
                datas.append([uid, close[1], url, 4])
            elif close[4] == 1:
                datas.append([uid, close[1], url, 5])
            else:
                if close[3] == 1:
                    datas.append([uid, close[1], url, 2])
                else:
                    datas.append([uid, close[1], url, 3])
        
        comments = mysql.get_comment_for_url(url)
        for comment in comments:
            uid = mysql.get_user_id(comment[0])[0][0]
            datas.append([uid, comment[1], url, 1])

        issue_prs = mysql.get_issue_pr_for_pr(url)
        for issue_pr in issue_prs:
            uid = mysql.get_user_id(issue_pr[0])[0][0]
            if issue_pr[3] == 1:
                datas.append([uid, issue_pr[1], url, 8])
            else:
                datas.append([uid, issue_pr[1], url, 9])

        labels = mysql.get_label_for_url(url)
        for label in labels:
            uid = mysql.get_user_id(label[0])[0][0]
            if label[3] == 0:
                datas.append([uid, label[1], url, 10])
            else:
                datas.append([uid, label[1], url, 11])

        mentions = mysql.get_mention_for_url(url)
        for mention in mentions:
            uid = mysql.get_user_id(mention[0])[0][0]
            datas.append([uid, mention[1], url, 12])
        
        commits = mysql.get_commit_for_url(url)
        for commit in commits:
            uid = mysql.get_user_id(commit[0])[0][0]
            datas.append([uid, commit[1], url, 1])

        merges = mysql.get_merge_for_url(url)
        for merge in merges:
            uid = mysql.get_user_id(merge[0])[0][0]
            datas.append([uid, merge[1], url, 13])

        reviews = mysql.get_review_for_url(url)
        for review in reviews:
            uid = mysql.get_user_id(review[0])[0][0]
            if review[3] == 1:
                datas.append([uid, review[1], url, 16])
            else:
                datas.append([uid, review[1], url, 15])


        if len(datas) < 6:
            return None

        datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
        for record in datas:
            uid = record[0]
            if uid not in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(record[3])

        res = []
        
        for key, value in user_dict.items():
            if(len(value) < 2):
                continue
            arr = [key] + [x + 1 for x in value]
            arr = [str(x) for x in arr]
            f.write(" ".join(arr))
            f.write("\n")
            # for i in range(1, len(value)):
            #     if(i - 5 < 0):
            #         res.append([key, [0] * (5 - i) + value[0:i], value[i]])
            #     else:
            #         res.append([key, value[i-5:i], value[i]])
        return res

    f = open(r'dataset\OSS\oss.txt', 'w')

    mysql = MySQL()
    datas = []
    issues = mysql.get_all_issues()
    prev = 0
    for i in range(len(issues)):
        percent = int(i * 100 / len(issues))
        if(percent != prev):
            print(percent)
            prev = percent
        issue = issues[i]
        # print(issue)
        issue_datas = generate_raw_sequential_record_from_issue(issue, mysql, f)
        # if(issue_datas != None):
        #     datas.extend(issue_datas)

    prs = mysql.get_all_prs()
    prev = 0
    for i in range(len(prs)):
        percent = int(i * 100 / len(prs))
        if(percent != prev):
            print(percent)
            prev = percent
        pr = prs[i]
        # print(issue)
        pr_datas = generate_sequential_record_from_pr(pr, mysql, f)
        # if(pr_datas != None):
        #     datas.extend(pr_datas)

    # df = pd.DataFrame(datas)
    # df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\squential_oss.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas



'''
    Behavior Type:
    - issue 17
    - comment 1
    - close 2
    - reopen 3
    - lock 4
    - unlock 5
    - pr 6
    - commit 7
    - link 8
    - unlink 9
    - label 10
    - unlabel 11
    - mention 12
    - merge 13
    - quote 14
    - review 15
    - dismiss 16
'''
def generate_sequential_dataset():
    mysql = MySQL()
    datas = []
    issues = mysql.get_all_issues()
    prev = 0
    for i in range(len(issues)):
        percent = int(i * 100 / len(issues))
        if(percent != prev):
            print(percent)
            prev = percent
        issue = issues[i]
        # print(issue)
        issue_datas = generate_sequential_record_from_issue(issue, mysql)
        if(issue_datas != None):
            datas.extend(issue_datas)

    prs = mysql.get_all_prs()
    prev = 0
    for i in range(len(prs)):
        percent = int(i * 100 / len(prs))
        if(percent != prev):
            print(percent)
            prev = percent
        pr = prs[i]
        # print(issue)
        pr_datas = generate_sequential_record_from_pr(pr, mysql)
        if(pr_datas != None):
            datas.extend(pr_datas)

    df = pd.DataFrame(datas)
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\squential_oss.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas

def generate_dataset_with_user_history():
    mysql = MySQL()
    datas = [] 

    issues = mysql.get_all_issues()
    prev = 0
    for i in range(len(issues)):
        percent = int(i * 100 / len(issues))
        if(percent != prev):
            print(percent)
            prev = percent
        issue = issues[i]
        # print(issue)
        issue_datas = generate_from_issue_with_user_history(issue, mysql)
        if(issue_datas != None):
            datas.extend(issue_datas)

    prs = mysql.get_all_prs()
    prev = 0
    for i in range(len(prs)):
        percent = int(i * 100 / len(prs))
        if(percent != prev):
            print(percent)
            prev = percent
        pr = prs[i]
        # print(issue)
        pr_datas = generate_from_pr_with_user_history(pr, mysql)
        if(pr_datas != None):
            datas.extend(pr_datas)


    df = pd.DataFrame(datas)
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss_with_user_history.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas

def generate_sequential_record_from_pr(pr, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_dict = {}
    url = pr[3]
    pr_uid = mysql.get_user_id(pr[2])[0][0]
    datas.append([pr_uid, pr[5], url, 6])

    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_pr(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        datas.append([uid, mention[1], url, 12])
    
    commits = mysql.get_commit_for_url(url)
    for commit in commits:
        uid = mysql.get_user_id(commit[0])[0][0]
        datas.append([uid, commit[1], url, 1])

    merges = mysql.get_merge_for_url(url)
    for merge in merges:
        uid = mysql.get_user_id(merge[0])[0][0]
        datas.append([uid, merge[1], url, 13])

    reviews = mysql.get_review_for_url(url)
    for review in reviews:
        uid = mysql.get_user_id(review[0])[0][0]
        if review[3] == 1:
            datas.append([uid, review[1], url, 16])
        else:
            datas.append([uid, review[1], url, 15])


    if len(datas) < 6:
        return None

    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for record in datas:
        uid = record[0]
        if uid not in user_dict:
            user_dict[uid] = []
        user_dict[uid].append(record[3])

    res = []
    
    for key, value in user_dict.items():
        for i in range(1, len(value)):
            if(i - 5 < 0):
                res.append([key, [0] * (5 - i) + value[0:i], value[i]])
            else:
                res.append([key, value[i-5:i], value[i]])
    return res

def generate_sequential_record_from_issue(issue, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_dict = {}
    url = issue[1]
    issue_uid = mysql.get_user_id(issue[3])[0][0]
    datas.append([issue_uid, issue[5], url, 17])
    
    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_issue(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        datas.append([uid, mention[1], url, 12])
    

    if len(datas) < 6:
        return None
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for record in datas:
        uid = record[0]
        if uid not in user_dict:
            user_dict[uid] = []
        user_dict[uid].append(record[3])

    res = []
    
    for key, value in user_dict.items():
        for i in range(1, len(value)):
            if(i - 5 < 0):
                res.append([key, [0] * (5 - i) + value[0:i], value[i]])
            else:
                res.append([key, value[i-5:i], value[i]])
    return res

def generate_dataset():
    mysql = MySQL()
    users = mysql.get_all_users()
    issues = mysql.get_all_issues()

    datas = [] 
    prev = 0
    for i in range(len(issues)):
        percent = int(i * 100 / len(issues))
        if(percent != prev):
            print(percent)
            prev = percent
        issue = issues[i]
        # print(issue)
        issue_datas = generate_from_issue(issue, mysql)
        if(issue_datas != None):
            datas.extend(issue_datas)

    df = pd.DataFrame(datas)
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas

def generate_sr_dataset():
    mysql = MySQL()
    users = mysql.get_all_users()

    datas = [] 
    prev = 0
    for i in range(len(users)):
        percent = int(i * 100 / len(users))
        if(percent != prev):
            print(percent)
            prev = percent
        user = users[i]
        # print(issue)
        user_datas = generate_from_username(user[0], mysql)
        if(user_datas != None):
            datas.extend(user_datas)

    df = pd.DataFrame(datas)
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss_sr.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas

def generate_dataset_from_pr():
    mysql = MySQL()
    prs = mysql.get_all_prs()

    datas = [] 
    prev = 0
    for i in range(len(prs)):
        percent = int(i * 100 / len(prs))
        if(percent != prev):
            print(percent)
            prev = percent
        pr = prs[i]
        # print(issue)
        issue_datas = generate_from_pr(pr, mysql)
        if(issue_datas != None):
            datas.extend(issue_datas)

    df = pd.DataFrame(datas)
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss_pr.csv')
    # df = pd.read_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv', header=0, usecols=[1, 2, 3, 4, 5])
    # print(df)
    return datas

def generate_from_pr(pr, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_set = set()
    url = pr[3]
    pr_uid = mysql.get_user_id(pr[2])[0][0]
    user_set.add(pr_uid)
    datas.append([pr_uid, pr[5], url, 6])

    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        user_set.add(uid)
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        user_set.add(uid)
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_pr(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        user_set.add(uid)
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        user_set.add(uid)
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        user_set.add(uid)
        datas.append([uid, mention[1], url, 12])
    
    commits = mysql.get_commit_for_url(url)
    for commit in commits:
        uid = mysql.get_user_id(commit[0])[0][0]
        user_set.add(uid)
        datas.append([uid, commit[1], url, 1])

    merges = mysql.get_merge_for_url(url)
    for merge in merges:
        uid = mysql.get_user_id(merge[0])[0][0]
        user_set.add(uid)
        datas.append([uid, merge[1], url, 13])

    reviews = mysql.get_review_for_url(url)
    for review in reviews:
        uid = mysql.get_user_id(review[0])[0][0]
        user_set.add(uid)
        if review[3] == 1:
            datas.append([uid, review[1], url, 16])
        else:
            datas.append([uid, review[1], url, 15])


    if len(datas) < 6:
        return None

    individual_records = {}
    for i in range(len(datas)):
        if datas[i][0] not in individual_records:
            individual_records[datas[i][0]] = np.ones(CONSTANT.TOTAL_TYPE)
        individual_records[datas[i][0]][datas[i][3]] = 0

    for u in individual_records:
        tmp = individual_records[u] * (np.arange(CONSTANT.TOTAL_TYPE) + 1)
        individual_records[u] = tmp[tmp != 0] - 1

    user_dict = dict(zip(user_set, [0 for i in range(len(user_set))]))
    res = []
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for i in range(5, len(datas)):
        group_user, group_behavior, _ = make_enum(datas[:i])
        # group_user, group_behavior = make_statistics(datas[:i])
        tmp_dict = deepcopy(user_dict)
        for j in range(i, len(datas)):
            if tmp_dict[datas[j][0]] == 0:
                negative_sample = int(np.random.choice(individual_records[datas[j][0]]))
                res.append([json.dumps(group_user), json.dumps(group_behavior), datas[j][0], datas[j][3], negative_sample])
                tmp_dict[datas[j][0]] = 1
    return res

def generate_from_pr_with_user_history(pr, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_set = set()
    url = pr[3]
    pr_uid = mysql.get_user_id(pr[2])[0][0]
    user_set.add(pr_uid)
    datas.append([pr_uid, pr[5], url, 6])

    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        user_set.add(uid)
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        user_set.add(uid)
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_pr(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        user_set.add(uid)
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        user_set.add(uid)
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        user_set.add(uid)
        datas.append([uid, mention[1], url, 12])
    
    commits = mysql.get_commit_for_url(url)
    for commit in commits:
        uid = mysql.get_user_id(commit[0])[0][0]
        user_set.add(uid)
        datas.append([uid, commit[1], url, 1])

    merges = mysql.get_merge_for_url(url)
    for merge in merges:
        uid = mysql.get_user_id(merge[0])[0][0]
        user_set.add(uid)
        datas.append([uid, merge[1], url, 13])

    reviews = mysql.get_review_for_url(url)
    for review in reviews:
        uid = mysql.get_user_id(review[0])[0][0]
        user_set.add(uid)
        if review[3] == 1:
            datas.append([uid, review[1], url, 16])
        else:
            datas.append([uid, review[1], url, 15])


    if len(datas) < 6:
        return None

    individual_records = {}
    for i in range(len(datas)):
        if datas[i][0] not in individual_records:
            individual_records[datas[i][0]] = np.ones(CONSTANT.TOTAL_TYPE)
        individual_records[datas[i][0]][datas[i][3]] = 0

    for u in individual_records:
        tmp = individual_records[u] * (np.arange(CONSTANT.TOTAL_TYPE) + 1)
        individual_records[u] = tmp[tmp != 0] - 1

    user_dict = dict(zip(user_set, [0 for i in range(len(user_set))]))
    res = []
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for i in range(5, len(datas)):
        group_user, group_behavior, group_dict = make_enum(datas[:i])
        # group_user, group_behavior = make_statistics(datas[:i])
        tmp_dict = deepcopy(user_dict)
        for j in range(i, len(datas)):
            if tmp_dict[datas[j][0]] == 0:
                if datas[j][0] in group_dict:
                    user_behavior = group_dict[datas[j][0]]
                    if len(user_behavior) < 5:
                        user_behavior = [0] * (5 - len(user_behavior)) + user_behavior
                    else:
                        user_behavior = user_behavior[-5:]
                    negative_sample = int(np.random.choice(individual_records[datas[j][0]]))
                    res.append([json.dumps(group_user), json.dumps(group_behavior), user_behavior, datas[j][0], datas[j][3], negative_sample])
                    tmp_dict[datas[j][0]] = 1
    return res

def generate_from_username(username, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    uid = mysql.get_user_id(username)[0][0]
    url=""
    closes = mysql.get_close_for_username(username)
    for close in closes:
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_username(username)
    for comment in comments:
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_username(url)
    for issue_pr in issue_prs:
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_username(username)
    for label in labels:
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_username(username)
    for mention in mentions:
        datas.append([uid, mention[1], url, 12])

    commits = mysql.get_commit_for_username(username)
    for commit in commits:
        datas.append([uid, commit[1], url, 1])

    merges = mysql.get_merge_for_username(username)
    for merge in merges:
        datas.append([uid, merge[1], url, 13])

    reviews = mysql.get_review_for_username(username)
    for review in reviews:
        if review[3] == 1:
            datas.append([uid, review[1], url, 16])
        else:
            datas.append([uid, review[1], url, 15])
    
    if len(datas) < 6:
        return None
    
    individual_records = np.ones(CONSTANT.TOTAL_TYPE)
    # 把做过的事置为0
    for i in range(len(datas)):
        individual_records[datas[i][3]] = 0

    # 过滤到为0的，剩下就是不为0的，然后从中选择负样本

    tmp = individual_records * (np.arange(CONSTANT.TOTAL_TYPE) + 1)
    individual_records = tmp[tmp != 0] - 1
    df = pd.DataFrame(datas)
    df.columns = ['uid', 'datetime', 'url', 'type']
    behavior_history = df.type.values.tolist()
    res = []
    
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for i in range(5, len(datas)):
        negative_sample = int(np.random.choice(individual_records))
        res.append([uid, behavior_history[i-5:i], datas[i][3], negative_sample])
    return res

def generate_from_issue(issue, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_set = set()
    url = issue[1]
    issue_uid = mysql.get_user_id(issue[3])[0][0]
    user_set.add(issue_uid)
    datas.append([issue_uid, issue[5], url, 0])

    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        user_set.add(uid)
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        user_set.add(uid)
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_issue(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        user_set.add(uid)
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        user_set.add(uid)
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        user_set.add(uid)
        datas.append([uid, mention[1], url, 12])
    

    if len(datas) < 6:
        return None

    individual_records = {}
    # 把做过的事置为0
    for i in range(len(datas)):
        if datas[i][0] not in individual_records:
            individual_records[datas[i][0]] = np.ones(CONSTANT.TOTAL_TYPE)
        individual_records[datas[i][0]][datas[i][3]] = 0

    # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
    for u in individual_records:
        tmp = individual_records[u] * (np.arange(CONSTANT.TOTAL_TYPE) + 1)
        individual_records[u] = tmp[tmp != 0] - 1

    user_dict = dict(zip(user_set, [0 for i in range(len(user_set))]))
    res = []
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for i in range(5, len(datas)):
        group_user, group_behavior, _ = make_enum(datas[:i])
        # group_user, group_behavior = make_statistics(datas[:i])
        tmp_dict = deepcopy(user_dict)
        for j in range(i, len(datas)):
            if tmp_dict[datas[j][0]] == 0:
                negative_sample = int(np.random.choice(individual_records[datas[j][0]]))
                res.append([json.dumps(group_user), json.dumps(group_behavior), datas[j][0], datas[j][3], negative_sample])
                tmp_dict[datas[j][0]] = 1
    return res

def generate_from_issue_with_user_history(issue, mysql: MySQL):
    '''
        uid
        time
        url
        type
    '''
    datas = []
    user_set = set()
    url = issue[1]
    issue_uid = mysql.get_user_id(issue[3])[0][0]
    user_set.add(issue_uid)
    datas.append([issue_uid, issue[5], url, 0])

    closes = mysql.get_close_for_url(url)
    for close in closes:
        uid = mysql.get_user_id(close[0])[0][0]
        user_set.add(uid)
        if close[4] == 0:
            datas.append([uid, close[1], url, 4])
        elif close[4] == 1:
            datas.append([uid, close[1], url, 5])
        else:
            if close[3] == 1:
                datas.append([uid, close[1], url, 2])
            else:
                datas.append([uid, close[1], url, 3])
    
    comments = mysql.get_comment_for_url(url)
    for comment in comments:
        uid = mysql.get_user_id(comment[0])[0][0]
        user_set.add(uid)
        datas.append([uid, comment[1], url, 1])

    issue_prs = mysql.get_issue_pr_for_issue(url)
    for issue_pr in issue_prs:
        uid = mysql.get_user_id(issue_pr[0])[0][0]
        user_set.add(uid)
        if issue_pr[3] == 1:
            datas.append([uid, issue_pr[1], url, 8])
        else:
            datas.append([uid, issue_pr[1], url, 9])

    labels = mysql.get_label_for_url(url)
    for label in labels:
        uid = mysql.get_user_id(label[0])[0][0]
        user_set.add(uid)
        if label[3] == 0:
            datas.append([uid, label[1], url, 10])
        else:
            datas.append([uid, label[1], url, 11])

    mentions = mysql.get_mention_for_url(url)
    for mention in mentions:
        uid = mysql.get_user_id(mention[0])[0][0]
        user_set.add(uid)
        datas.append([uid, mention[1], url, 12])
    

    if len(datas) < 6:
        return None

    individual_records = {}
    # 把做过的事置为0
    for i in range(len(datas)):
        if datas[i][0] not in individual_records:
            individual_records[datas[i][0]] = np.ones(CONSTANT.TOTAL_TYPE)
        individual_records[datas[i][0]][datas[i][3]] = 0

    # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
    for u in individual_records:
        tmp = individual_records[u] * (np.arange(CONSTANT.TOTAL_TYPE) + 1)
        individual_records[u] = tmp[tmp != 0] - 1

    user_dict = dict(zip(user_set, [0 for i in range(len(user_set))]))
    res = []
    datas.sort(key=functools.cmp_to_key(lambda a, b: 1 if a[1].__ge__(b[1]) else -1))
    for i in range(5, len(datas)):
        group_user, group_behavior, group_dict = make_enum(datas[:i])
        # group_user, group_behavior = make_statistics(datas[:i])
        tmp_dict = deepcopy(user_dict)
        for j in range(i, len(datas)):
            if tmp_dict[datas[j][0]] == 0:
                if datas[j][0] in group_dict:
                    user_behavior = group_dict[datas[j][0]]
                    if len(user_behavior) < 5:
                        user_behavior = [0] * (5 - len(user_behavior)) + user_behavior
                    else:
                        user_behavior = user_behavior[-5:]
                    negative_sample = int(np.random.choice(individual_records[datas[j][0]]))
                    
                    res.append([json.dumps(group_user), json.dumps(group_behavior), user_behavior, datas[j][0], datas[j][3], negative_sample])
                    tmp_dict[datas[j][0]] = 1
    return res


def make_enum(data_slice):
    group_user_behavior = {}
    group_user = []
    group_behavior = []
    group_dict = {}

    for data in data_slice:
        if data[0] not in group_user_behavior:
            group_user_behavior[data[0]] = {}
            group_dict[data[0]] = []
        if data[3] not in group_user_behavior[data[0]]:
            group_user_behavior[data[0]][data[3]] = 0
        group_user_behavior[data[0]][data[3]] += 1
        group_dict[data[0]].append(data[3])
    for k, v in group_user_behavior.items():
        group_user.append(k)
        group_behavior.append(v)

    return group_user, group_behavior, group_dict

def make_statistics(data_slice):
    group_user = np.zeros(CONSTANT.TOTAL_USER)
    group_behavior = np.zeros([CONSTANT.TOTAL_USER, CONSTANT.TOTAL_TYPE])
    for data in data_slice:
        group_user[data[0]] = 1
        group_behavior[data[0]][data[3]] += 1
    return group_user.tolist(), group_behavior.tolist()

def union_dataset():
    issue_df = pd.read_csv(r"C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss_issue.csv", header=0, usecols=[1,2,3,4,5])
    pr_df = pd.read_csv(r"C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss_pr.csv", header=0, usecols=[1,2,3,4,5])

    df = pd.concat([issue_df, pr_df])
    df.to_csv(r'C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv')

if __name__ == "__main__":

    generate_sequential_dataset()
    # generate_dataset_with_user_history()
    # generate_sr_dataset()
    # from GPR.model.TorchDataset.GroupDataset import GroupDataset
    # from torch.utils.data import DataLoader, SubsetRandomSampler
    # from GPR.model.GPR.GPR import GPR
    # import GPR.config.GPR as GPR_CONFIG

    # generate_dataset_from_pr()
    # # datas = generate_dataset_from_pr()
    # # import os
    # # os._exit(0)
    # # shuffled_indices = np.random.permutation(len(datas))
    # # train_idx = shuffled_indices[:int(0.8*len(datas))]
    # # val_idx = shuffled_indices[int(0.8*len(datas)):]

    # # dataset = GroupDataset(datas)
    # # train_dataloader = DataLoader(dataset=dataset, batch_size=1)

    # # model = GPR(GPR_CONFIG.USE_NUM, GPR_CONFIG.ITEM_NUM, GPR_CONFIG.USER_EMB_SIZE, GPR_CONFIG.ITEM_EMB_SIZE)
    
    # # for i, (group_user, group_behavior, target_user, target_behavior, negative_behavior) in enumerate(train_dataloader):
    # #     model(group_user, group_behavior, target_user, target_behavior)

    # df = pd.read_csv(r"C:\Neil\Code\internet_of_behavior\pyiob\dataset\OSS\oss.csv", header=0, usecols=[1,2,3,4,5])
    # dataset = GroupDataset(df.values, CONSTANT.TOTAL_USER, CONSTANT.TOTAL_TYPE)
    # train_dataloader = DataLoader(dataset=dataset, batch_size=1)
    # model = GPR(GPR_CONFIG.USE_NUM, GPR_CONFIG.ITEM_NUM, GPR_CONFIG.USER_EMB_SIZE, GPR_CONFIG.ITEM_EMB_SIZE)
    # for i, (group_user, group_behavior, target_user, target_behavior, negative_behavior) in enumerate(train_dataloader):
    #     model(group_user, group_behavior, target_user, target_behavior)