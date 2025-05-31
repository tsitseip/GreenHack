import pickle

speeds = {
    'Car' : 80,
    'Bus' : 80,
    'Airplane' : 800,
    'Excl.' : 80,
    'Train' : 90,
    'Public mass transport' : 50
}

prices = {
    'Car' : 5.5,
    'Bus' : 2,
    'Airplane' : 4.5,
    'Train' : 2,
    'Public mass transport' : 1,
    'Excl.' : 0
}

#(from, to, (emission,time,cost))
#13 = transport
#5 = to
#7 = from
#last = 26
#time = 25/speeds[transport]
#cost = prices[transport]*25

graph = []

with open('dictEm.pkl', 'rb') as fp:
    cachedDict = pickle.load(fp)
    for i in range(len(cachedDict['Subsidiary'])):
        transport = cachedDict['Adjusted transport'][i][0]
        if float(cachedDict['Emission'][i][0]) == 0 or float(cachedDict['PA km'][i][0]) == 0:
            continue
        edge = {
            'start' : cachedDict['Trip departure'][i][0],
            'end' : cachedDict['Location'][i][0],
            'weight' : (float(cachedDict['Emission'][i][0]), float(cachedDict['PA km'][i][0])/speeds[transport], float(cachedDict['PA km'][i][0])*prices[transport]),
            'transport' : transport
        }
        graph.append(edge)

with open('graph.pkl', 'wb') as fp:
    pickle.dump(graph, fp)

