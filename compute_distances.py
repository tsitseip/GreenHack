import heapq,pickle
import random
from collections import defaultdict

with open('cities.pkl', 'rb') as w:
    cities=pickle.load(w)

class Weight:
    def __init__(self, weight:tuple):
        self.weight = weight
    def __lt__(self, other):
        value = self.weight[0]*40 + self.weight[1] * 6 + self.weight[2] * 0.001
        value1=other.weight[0]*40 + other.weight[1] * 6 + other.weight[2] * 0.001
        return value<value1
    def __eq__(self, other):
        return self.weight == other.weight
    def __str__(self):
        return str(self.weight)
# emissions, time, price

def compute_distances(point1:str, point2:str, graph_edges:list,k:int):
    # Priority queue: (distance, vertex)
    #
    graph_adjacency=defaultdict(list)
    for edge in graph_edges:
        start = edge['start']
        end = edge['end']
        if start not in graph_adjacency:
            graph_adjacency[start]=[]
        if end not in graph_adjacency:
            graph_adjacency[end] = []
        graph_adjacency[start].append((edge['end'],Weight(edge['weight'])))
    #

    pq = [(Weight((0,0,0)), point1 , [point1])]
    # Distances dictionary
    distances = {vertex: [(Weight((float('inf'),float('inf'),float('inf'))),[])] for vertex in cities}
    distances[point1] = [(Weight((0,0,0)),[point1])]
    # Visited set
    visited = set()
    while pq:
        current_distance, current_vertex, path = heapq.heappop(pq)
        if current_vertex in visited:
            continue
        visited.add(current_vertex)
        for neighbor, weight in graph_adjacency[current_vertex]:
            distance = Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight)))
            if len(distances[neighbor])<k or distance < distances[neighbor][-1][0]:
                distances[neighbor].append((Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight))),path+[neighbor]))
                distances[neighbor] = sorted(distances[neighbor])
                if len(distances[neighbor])>k:
                    distances[neighbor].pop(-1)
                heapq.heappush(pq, (distance, neighbor, path + [neighbor]))
    return list(map(lambda x: x[1]+[str(x[0])],distances[point2]))

with open('graph.pkl', 'rb') as fp:
        with open('test_dataset.pkl','wb') as write:
            graph_dict=pickle.load(fp)
            llst=list(cities)
            final = []
            for i in range(50):
                start=random.choice(llst)
                end=random.choice(llst)
                print(start,':',end)
                lst=compute_distances(start,end,graph_dict,5)
                for ls in lst:
                    final.append(ls[-1])
            pickle.dump(final,write)