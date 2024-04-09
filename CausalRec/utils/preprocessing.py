import pandas as pd
import numpy as np
import math
import copy

'''
乐器
'''
def music_instrument_5_to_10(df: pd.DataFrame):
    reviewers = list(set(df["reviewerID"]))
    print(len(reviewers))
    res = pd.DataFrame(columns=list(df))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        if type(reviewer) == float and math.isnan(reviewer):
            continue
        tmp_df = df.loc[df["reviewerID"] == reviewer]
        if(tmp_df.shape[0] > 9):
            res = pd.concat([res, tmp_df])

    res.columns = list(df)
    df = res

    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    print(len(reviewers))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        df.loc[df[df['reviewerID'] == reviewer].index, 'reviewerID']= i
    
    print(len(items))
    for i in range(len(items)):
        if (i % 100 == 0):
            print(i)
        item = items[i]
        df.loc[df[df['asin'] == item].index, 'asin']= i+1

    df.to_csv("CausalRec/dataset/Musical_Instruments_10.csv", index=False)
        
    # res.to_csv("dataset\Musical_Instruments_10.csv", header=list(df),  index=False)

def music_instrument_input():
    path = "CausalRec/dataset/Musical_Instruments_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(1, len(asins)):
            if(i - 5 < 0):
                datas.append([reviewer, [0] * (5 - i) + asins[0:i], asins[i]])
            else:
                datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def music_instrument_input_of_rnn():
    path = "CausalRec/dataset/Musical_Instruments_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(5, len(asins)):
            datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def music_instrument_input_of_bert(length = 5):
    path = "CausalRec/dataset/Musical_Instruments_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        sorted_behaviors = behaviors.sort_values(by="unixReviewTime")
        asins = sorted_behaviors.asin.values.tolist()
        overall = sorted_behaviors.overall.values.tolist()
        reviewTime = sorted_behaviors.unixReviewTime.values.astype(float)
        
        beforeTime = copy.deepcopy(reviewTime)
        beforeTime = np.insert(beforeTime, 0, reviewTime[0])
        beforeTime = np.delete(beforeTime, -1)
        reviewTimeDiff = (reviewTime - beforeTime) / 3600 / 24

        for i in range(len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i]])
    return datas, len(items), len(reviewers)

def music_instrument_input_of_nova(length = 5):
    path = "CausalRec/dataset/Musical_Instruments_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    df.replace(np.nan, 0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        sorted_behaviors = behaviors.sort_values(by="unixReviewTime")
        asins = sorted_behaviors.asin.values.tolist()
        overall = sorted_behaviors.overall.values.tolist()
        reviewTime = sorted_behaviors.unixReviewTime.values.astype(float)
        
        beforeTime = copy.deepcopy(reviewTime)
        beforeTime = np.insert(beforeTime, 0, reviewTime[0])
        beforeTime = np.delete(beforeTime, -1)
        reviewTimeDiff = ((reviewTime - beforeTime) / 3600 / 24).tolist()

        for i in range(1, len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i], [0] * (length - i) + overall[0:i], [0] * (length - i) + reviewTimeDiff[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i], overall[i-length:i], reviewTimeDiff[i-length:i]])
    return datas, len(items), len(reviewers)

'''
音乐
'''
def digital_music_5_to_10(df: pd.DataFrame):
    reviewers = list(set(df["reviewerID"]))
    print(len(reviewers))
    res = pd.DataFrame(columns=list(df))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        if type(reviewer) == float and math.isnan(reviewer):
            continue
        tmp_df = df.loc[df["reviewerID"] == reviewer]
        if(tmp_df.shape[0] > 9):
            res = pd.concat([res, tmp_df])

    res.columns = list(df)
    df = res

    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    print(len(reviewers))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        df.loc[df[df['reviewerID'] == reviewer].index, 'reviewerID']= i
    
    print(len(items))
    for i in range(len(items)):
        if (i % 100 == 0):
            print(i)
        item = items[i]
        df.loc[df[df['asin'] == item].index, 'asin']= i+1

    df.to_csv("CausalRec/dataset/Digital_Music_10.csv", index=False)

def digital_music_input():
    path = "CausalRec/dataset/Digital_Music_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(1, len(asins)):
            if(i - 5 < 0):
                datas.append([reviewer, [0] * (5 - i) + asins[0:i], asins[i]])
            else:
                datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def digital_music_input_of_rnn():
    path = "CausalRec/dataset/Digital_Music_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(5, len(asins)):
            datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def digital_music_input_of_bert(length = 5):
    path = "CausalRec/dataset/Digital_Music_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i]])
    return datas, len(items), len(reviewers)

def digital_music_input_of_nova(length = 5):
    path = "CausalRec/dataset/Digital_Music_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        sorted_behaviors = behaviors.sort_values(by="unixReviewTime")
        asins = sorted_behaviors.asin.values.tolist()
        overall = sorted_behaviors.overall.values.tolist()
        reviewTime = sorted_behaviors.unixReviewTime.values.astype(float)
        
        beforeTime = copy.deepcopy(reviewTime)
        beforeTime = np.insert(beforeTime, 0, reviewTime[0])
        beforeTime = np.delete(beforeTime, -1)
        reviewTimeDiff = ((reviewTime - beforeTime) / 3600 / 24).tolist()

        for i in range(1, len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i], [0] * (length - i) + overall[0:i], [0] * (length - i) + reviewTimeDiff[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i], overall[i-length:i], reviewTimeDiff[i-length:i]])
    return datas, len(items), len(reviewers)

'''
化妆品
'''
def luxury_beauty_5_to_10(df: pd.DataFrame):
    reviewers = list(set(df["reviewerID"]))
    print(len(reviewers))
    res = pd.DataFrame(columns=list(df))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        if type(reviewer) == float and math.isnan(reviewer):
            continue
        tmp_df = df.loc[df["reviewerID"] == reviewer]
        if(tmp_df.shape[0] > 9):
            res = pd.concat([res, tmp_df])

    res.columns = list(df)
    df = res

    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    print(len(reviewers))
    for i in range(len(reviewers)):
        if (i % 100 == 0):
            print(i)
        reviewer = reviewers[i]
        df.loc[df[df['reviewerID'] == reviewer].index, 'reviewerID']= i
    
    print(len(items))
    for i in range(len(items)):
        if (i % 100 == 0):
            print(i)
        item = items[i]
        df.loc[df[df['asin'] == item].index, 'asin']= i+1

    df.to_csv("CausalRec/dataset/Luxury_Beauty_10.csv", index=False)

def luxury_beauty_input():
    path = "CausalRec/dataset/Luxury_Beauty_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(1, len(asins)):
            if(i - 5 < 0):
                datas.append([reviewer, [0] * (5 - i) + asins[0:i], asins[i]])
            else:
                datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def luxury_beauty_input_of_rnn():
    path = "CausalRec/dataset/Luxury_Beauty_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(5, len(asins)):
            datas.append([reviewer, asins[i-5:i], asins[i]])
    return datas, len(items), len(reviewers)

def luxury_beauty_input_of_bert(length = 5):
    path = "CausalRec/dataset/Luxury_Beauty_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        asins = behaviors.sort_values(by="unixReviewTime").asin.values.tolist()

        for i in range(len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i]])
    return datas, len(items), len(reviewers)

def luxury_beauty_input_of_nova(length = 5):
    path = "CausalRec/dataset/Luxury_Beauty_10.csv"
    df: pd.DataFrame = pd.read_csv(path, header=0)
    df.replace(np.nan, 0)
    reviewers = list(set(df["reviewerID"]))
    items = list(set(df["asin"]))
    datas = []
    for reviewer in reviewers:
        # 排序
        behaviors = df.loc[df["reviewerID"] == reviewer]
        sorted_behaviors = behaviors.sort_values(by="unixReviewTime")
        asins = sorted_behaviors.asin.values.tolist()
        overall = sorted_behaviors.overall.values.tolist()
        reviewTime = sorted_behaviors.unixReviewTime.values.astype(float)
        
        beforeTime = copy.deepcopy(reviewTime)
        beforeTime = np.insert(beforeTime, 0, reviewTime[0])
        beforeTime = np.delete(beforeTime, -1)
        reviewTimeDiff = ((reviewTime - beforeTime) / 3600 / 24).tolist()

        for i in range(1, len(asins)+1):
            if(i - length < 0):
                datas.append([reviewer, [0] * (length - i) + asins[0:i], [0] * (length - i) + overall[0:i], [0] * (length - i) + reviewTimeDiff[0:i]])
            else:
                datas.append([reviewer, asins[i-length:i], overall[i-length:i], reviewTimeDiff[i-length:i]])
    return datas, len(items), len(reviewers)



if __name__ == "__main__":
    cnt = 0
    rows = 0
    lines = open(r'ssss').readlines()
    for line in lines:
        rows += 1
        items = line.strip().split(' ')
        cnt += len(items)

    print((cnt-rows) / rows)