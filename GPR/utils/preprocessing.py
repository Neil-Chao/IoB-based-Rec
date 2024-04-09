import pandas as pd
import numpy as np
import json

def process_meetup(df: pd.DataFrame):

    groups = list(set(df["group_id"]))
    print(len(groups))
    for i in range(len(groups)):
        if (i % 100 == 0):
            print(i)
        group = groups[i]
        df.loc[df[df['group_id'] == group].index, 'group_id']= i+1

    events = list(set(df["event_id"]))
    print(len(events))
    for i in range(len(events)):
        if (i % 100 == 0):
            print(i)
        event = events[i]
        df.loc[df[df['event_id'] == event].index, 'event_id']= i+1

    members = list(set(df["member_id"]))
    print(len(members))
    for i in range(len(members)):
        if (i % 100 == 0):
            print(i)
        member = members[i]
        df.loc[df[df['member_id'] == member].index, 'member_id']= i+1

    df.to_csv("gpr.csv", index=False)

def generate_meetup_group_dataset(df: pd.DataFrame):
    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[2] not in group_user_behavior:
                group_user_behavior[data[2]] = {}
                group_dict[data[2]] = []
            if data[1] not in group_user_behavior[data[2]]:
                group_user_behavior[data[2]][data[1]] = 0
            group_user_behavior[data[2]][data[1]] += 1
            group_dict[data[2]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict

    groups = list(set(df["group_id"]))
    print(len(groups))

    events = list(set(df["event_id"]))
    event_num = len(events)

    members = list(set(df["member_id"]))
    member_num = len(members)

    output = []

    for i in range(len(groups)):
        if (i % 100 == 0):
            print(i)
        group = groups[i]
        datas = df[df['group_id'] == group].values.tolist()

        if len(datas) < 6:
            continue

        individual_records = {}
        # 把做过的事置为0
        for i in range(len(datas)):
            if datas[i][2] not in individual_records:
                # 保证负采样时不选中0行为
                tmp_ones = np.ones(event_num)
                # tmp_ones[0] = 0
                individual_records[datas[i][2]] = tmp_ones
            individual_records[datas[i][2]][datas[i][1]-1] = 0

        # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
        for u in individual_records:
            tmp = individual_records[u] * (np.arange(event_num) + 1)
            individual_records[u] = tmp[tmp != 0]

        user_dict = dict(zip(members, [0 for i in range(member_num)]))
        res = []
        for i in range(5, len(datas)):
            _, _, group_dict = make_enum(datas[:i])

            if i >= 50:
                group_user, group_behavior, _ = make_enum(datas[i-50:i])
            else:
                group_user, group_behavior, _ = make_enum(datas[:i])
            if(datas[i][2] in group_dict):
                negative_sample = int(np.random.choice(individual_records[datas[i][2]]))
                user_behavior = group_dict[datas[i][2]]
                if len(user_behavior) < 5:
                    user_behavior = [0] * (5 - len(user_behavior)) + user_behavior
                else:
                    user_behavior = user_behavior[-5:]
                res.append([json.dumps(group_user), json.dumps(group_behavior), user_behavior, datas[i][2], datas[i][1], negative_sample])
        output.extend(res)
    df = pd.DataFrame(output)
    df.to_csv(r'new_group_meetup.csv')

def generate_meetup_cf_dataset(df: pd.DataFrame):

    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[2] not in group_user_behavior:
                group_user_behavior[data[2]] = {}
                group_dict[data[2]] = []
            if data[1] not in group_user_behavior[data[2]]:
                group_user_behavior[data[2]][data[1]] = 0
            group_user_behavior[data[2]][data[1]] += 1
            group_dict[data[2]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict
    
    groups = list(set(df["group_id"]))

    res = []

    for i in range(len(groups)):
        if (i % 100 == 0):
            print(i)
        group = groups[i]
        group_df = df[df['group_id'] == group]
        group_events = list(set(group_df["event_id"]))

        for j in range(len(group_events)):
            group_event = group_events[j]
            datas = group_df[group_df['event_id'] == group_event].values.tolist()
            group_user, group_behavior, group_dict = make_enum(datas)
            if len(group_user) > 1:
                res.append([json.dumps(group_user), group_event])

    df = pd.DataFrame(res)
    df.to_csv(r'group_meetup_for_cf.csv')
        
def generate_meetup_mrtransformer_dataset(df: pd.DataFrame):
    f = open(r'meetup.txt', 'w')
    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[2] not in group_user_behavior:
                group_user_behavior[data[2]] = {}
                group_dict[data[2]] = []
            if data[1] not in group_user_behavior[data[2]]:
                group_user_behavior[data[2]][data[1]] = 0
            group_user_behavior[data[2]][data[1]] += 1
            group_dict[data[2]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict

    groups = list(set(df["group_id"]))
    print(len(groups))

    events = list(set(df["event_id"]))
    event_num = len(events)

    members = list(set(df["member_id"]))
    member_num = len(members)

    output = []

    for i in range(len(groups)):
        if (i % 100 == 0):
            print(i)
        group = groups[i]
        datas = df[df['group_id'] == group].values.tolist()

        if len(datas) < 6:
            continue

        individual_records = {}
        # 把做过的事置为0
        for i in range(len(datas)):
            if datas[i][2] not in individual_records:
                # 保证负采样时不选中0行为
                tmp_ones = np.ones(event_num)
                # tmp_ones[0] = 0
                individual_records[datas[i][2]] = tmp_ones
            individual_records[datas[i][2]][datas[i][1]-1] = 0

        # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
        for u in individual_records:
            tmp = individual_records[u] * (np.arange(event_num) + 1)
            individual_records[u] = tmp[tmp != 0]

        if len(datas) > 5:
            _, _, group_dict = make_enum(datas)
            # group_user, group_behavior = make_statistics(datas[:i])
            for u in group_dict:
                user_behavior = group_dict[u]
                if len(user_behavior) > 1:
                    user_behavior = [str(x) for x in [u] + user_behavior]
                    f.write(" ".join(user_behavior))
                    f.write("\n")


def process_camra2011(df: pd.DataFrame):
    events = list(set(df["event_id"]))
    users = list(set(df["user_id"]))
    for i in range(len(events)):
        if (i % 100 == 0):
            print(i)
        event = events[i]
        df.loc[df[df['event_id'] == event].index, 'event_id']= i+1


    for i in range(len(users)):
        if (i % 100 == 0):
            print(i)
        user = users[i]
        df.loc[df[df['user_id'] == user].index, 'user_id']= i+1

    df.to_csv("new_gpr.csv", index=False)   

def generate_camra2011_group_dataset(df: pd.DataFrame, group_path):

    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[0] not in group_user_behavior:
                group_user_behavior[data[0]] = {}
                group_dict[data[0]] = []
            if data[1] not in group_user_behavior[data[0]]:
                group_user_behavior[data[0]][data[1]] = 0
            group_user_behavior[data[0]][data[1]] += 1
            group_dict[data[0]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict
    
    g_m_d = {}
    with open(group_path, 'r') as f:
        line = f.readline().strip()
        while line != None and line != "":
            a = line.split(' ', maxsplit=1)
            g = int(a[0])
            g_m_d[g] = []
            for m in a[1].split(','):
                g_m_d[g].append(int(m))
            line = f.readline().strip()
    
    group_num = len(g_m_d)

    events = list(set(df["event_id"]))
    event_num = len(events)

    members = list(set(df["user_id"]))
    member_num = len(members)

    output = []

    for group, group_members in g_m_d.items():
        datas = df[df['user_id'].isin(group_members)].values.tolist()

        if len(datas) < 6:
            continue

        individual_records = {}
        # 把做过的事置为0
        for i in range(len(datas)):
            if datas[i][0] not in individual_records:
                # 保证负采样时不选中0行为
                tmp_ones = np.ones(event_num)
                # tmp_ones[0] = 0
                individual_records[datas[i][0]] = tmp_ones
            individual_records[datas[i][0]][datas[i][1]-1] = 0

        # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
        for u in individual_records:
            tmp = individual_records[u] * (np.arange(event_num) + 1)
            individual_records[u] = tmp[tmp != 0]

        user_dict = dict(zip(members, [0 for i in range(member_num)]))
        res = []
        for i in range(5, len(datas)):
            group_user, group_behavior, group_dict = make_enum(datas[:i])

            if(datas[i][0] in group_dict):
                negative_sample = int(np.random.choice(individual_records[datas[i][0]]))
                user_behavior = group_dict[datas[i][0]]
                if len(user_behavior) < 5:
                    user_behavior = [0] * (5 - len(user_behavior)) + user_behavior
                else:
                    user_behavior = user_behavior[-5:]
                res.append([json.dumps(group_user), json.dumps(group_behavior), user_behavior, datas[i][0], datas[i][1], negative_sample])
        output.extend(res)
    df = pd.DataFrame(output)
    df.to_csv(r'new_group_CAMRa2011.csv')    

def generate_camra_cf_dataset(df: pd.DataFrame, group_path):

    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[0] not in group_user_behavior:
                group_user_behavior[data[0]] = {}
                group_dict[data[0]] = []
            if data[1] not in group_user_behavior[data[0]]:
                group_user_behavior[data[0]][data[1]] = 0
            group_user_behavior[data[0]][data[1]] += 1
            group_dict[data[0]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict
    
    g_m_d = {}
    with open(group_path, 'r') as f:
        line = f.readline().strip()
        while line != None and line != "":
            a = line.split(' ', maxsplit=1)
            g = int(a[0])
            g_m_d[g] = []
            for m in a[1].split(','):
                g_m_d[g].append(int(m))
            line = f.readline().strip()

    res = []

    for group, group_members in g_m_d.items():
        group_df = df[df['user_id'].isin(group_members)]
        group_datas = group_df.values.tolist()
        event_set = set()
        for group_data in group_datas:
            event_set.add(group_data[1])
        for event in event_set:
            datas = group_df[group_df['event_id'] == event].values.tolist()
            group_user, group_behavior, group_dict = make_enum(datas)
            if len(group_user) > 1:
                res.append([json.dumps(group_user), event])

    df = pd.DataFrame(res)
    df.to_csv(r'group_CAMRa2011_for_cf.csv')

def generate_camra2011_mrtransformer_dataset(df: pd.DataFrame, group_path):
    
    def make_enum(data_slice):
        group_user_behavior = {}
        group_user = []
        group_behavior = []
        group_dict = {}

        for data in data_slice:
            if data[0] not in group_user_behavior:
                group_user_behavior[data[0]] = {}
                group_dict[data[0]] = []
            if data[1] not in group_user_behavior[data[0]]:
                group_user_behavior[data[0]][data[1]] = 0
            group_user_behavior[data[0]][data[1]] += 1
            group_dict[data[0]].append(data[1])
        for k, v in group_user_behavior.items():
            group_user.append(k)
            group_behavior.append(v)

        return group_user, group_behavior, group_dict
    
    g_m_d = {}
    with open(group_path, 'r') as f:
        line = f.readline().strip()
        while line != None and line != "":
            a = line.split(' ', maxsplit=1)
            g = int(a[0])
            g_m_d[g] = []
            for m in a[1].split(','):
                g_m_d[g].append(int(m))
            line = f.readline().strip()
    
    group_num = len(g_m_d)

    events = list(set(df["event_id"]))
    event_num = len(events)

    members = list(set(df["user_id"]))
    member_num = len(members)

    output = []
    f = open(r'CAMRa2011.txt', 'w')
    for group, group_members in g_m_d.items():
        datas = df[df['user_id'].isin(group_members)].values.tolist()

        if len(datas) < 6:
            continue

        individual_records = {}
        # 把做过的事置为0
        for i in range(len(datas)):
            if datas[i][0] not in individual_records:
                # 保证负采样时不选中0行为
                tmp_ones = np.ones(event_num)
                # tmp_ones[0] = 0
                individual_records[datas[i][0]] = tmp_ones
            individual_records[datas[i][0]][datas[i][1]-1] = 0

        # 过滤到为0的，剩下就是不为0的，然后从中选择负样本
        for u in individual_records:
            tmp = individual_records[u] * (np.arange(event_num) + 1)
            individual_records[u] = tmp[tmp != 0]

        if len(datas) > 5:
            _, _, group_dict = make_enum(datas)
            # group_user, group_behavior = make_statistics(datas[:i])
            for u in group_dict:
                user_behavior = group_dict[u]
                if len(user_behavior) > 1:
                    user_behavior = [str(x) for x in [u] + user_behavior]
                    f.write(" ".join(user_behavior))
                    f.write("\n")


if __name__ == "__main__":
    df = pd.read_csv(r"new_gpr.csv", header=0)
    group_df = r"groupMember.txt"
    generate_camra2011_mrtransformer_dataset(df, group_df)