with open('data_parsed_em.csv', 'r') as f:
    line = f.readline()
    # print(line)
    columns = []
    name = ""
    started = False
    for c in line:
        if c=='\"':
            started = not started
        else:
            if started == True or c != ',':
                name += c
            else:
                if c==',':
                    columns.append(name)
                    name = ""
    if name != "":
        columns.append(name[:-1])
        name = ""
    # print(columns)

    dict = {}
    for i in columns:
        dict[i] = []

    for line in f:
        name = ""
        started = False
        i = 0
        for c in line:
            if c=='\"':
                started = not started
            else:
                if started == True or c != ',':
                    name += c
                else:
                    if c==',':
                        dict[columns[i]].append((name, i))
                        name = ""
                        i+=1
        if name != "":
            dict[columns[i]].append((name[:-1], i))
            name = ""

import pickle

with open('dict.pkl', 'wb') as fp:
    pickle.dump(dict, fp)


