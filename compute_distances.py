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
        graph_adjacency[start].append((edge['end'],Weight(edge['weight']),edge['transport']))
        graph_adjacency[end].append((start, Weight(edge['weight']),edge['transport']))
    #

    pq = [(Weight((0,0,0)), point1 , [point1])]
    # Distances dictionary
    distances=defaultdict(list)
    distances[point1] = [(Weight((0,0,0)),[point1])]
    # Visited set
    visited = set()
    visited.add(point2)
    while pq:
        current_distance, current_vertex, path = heapq.heappop(pq)
        if current_vertex in visited:
            continue
        visited.add(current_vertex)
        for neighbor, weight, transport in graph_adjacency[current_vertex]:
            distance = Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight)))
            if len(distances[neighbor])<k or distance < distances[neighbor][-1][0]:
                distances[neighbor].append((Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight))),path+[neighbor+'('+transport+')']))
                distances[neighbor] = sorted(distances[neighbor])
                if len(distances[neighbor])>k:
                    distances[neighbor].pop(-1)
                heapq.heappush(pq, (distance, neighbor, path + [neighbor+'('+transport+')']))
    return list(map(lambda x: x[1]+[x[0].weight],distances[point2]))


# with open('graph.pkl', 'rb') as fp:
#         with open('test_dataset.pkl','wb') as write:
#             graph_dict=pickle.load(fp)
#             llst=list(cities)
#             final = []
#             for i in range(20):
#                 lst = []
#                 while len(lst) != 5:
#                     start=random.choice(llst)
#                     end=random.choice(llst)
#                     print(i, ': ', start, ':', end)
#                     lst=compute_distances(start,end,graph_dict,5)
#                 fn = []
#                 for ls in lst:
#                     fn.append(ls[-1])
#                 final.append(fn)
#             pickle.dump(final,write)

