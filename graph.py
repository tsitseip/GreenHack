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

with open('dict_em.pkl', 'rb') as fp:
    cachedDict = pickle.load(fp)
    for trip in cachedDict:
        graph.append((trip[5][0], trip[7][0], (trip[26][0], trip[25][0]/speeds[trip[13][0]], prices[trip[13][0]])))


