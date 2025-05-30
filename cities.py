import pickle

cities = set()

with open('dictEm.pkl', 'rb') as fp:
    cachedDict = pickle.load(fp)
    for i in range(len(cachedDict['Subsidiary'])):
        cities.add(cachedDict['Location'][i])
        cities.add(cachedDict['Trip departure'][i])

with open('cities.pkl', 'wb') as fp:
    pickle.dump(cities, fp)