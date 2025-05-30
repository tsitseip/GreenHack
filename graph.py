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
        graph.append((cachedDict['Trip departure'][i][0], cachedDict['Location'][i][0], (cachedDict['Emission'][i][0], float(cachedDict['PA km'][i][0])/speeds[transport], prices[transport])))

with open('graph.pkl', 'wb') as fp:
    pickle.dump(list, fp)

