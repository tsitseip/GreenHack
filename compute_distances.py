import heapq
class Weight:
    def __init__(self, weight:tuple):
        self.weight = weight
    def __lt__(self, other):
        return sum(map(lambda x: x**2,*self.weight))<max(other.weight)
    def __eq__(self, other):
        return self.weight == other.weight
    def __str__(self):
        return str(self.weight)
# emissions, time, price

def compute_distances(point1:str, point2:str, graph_edges:list):
    # Priority queue: (distance, vertex)
    #
    graph_adjacency={}
    for edge in graph_edges:
        start = edge['start']
        end = edge['end']
        if start not in graph_adjacency:
            graph_adjacency[start]=[]
        if end not in graph_adjacency:
            graph_adjacency[end] = []
        graph_adjacency[start].append((edge['end'],Weight(edge['weight'])))
    #

    pq = [(Weight((0,0,0)), point1)]
    # Distances dictionary
    distances = {vertex: Weight((float('inf'),float('inf'),float('inf'))) for vertex in graph_adjacency.keys()}
    distances[point1] = Weight((0,0,0))
    # Visited set
    visited = set()
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_vertex in visited:
            continue
        visited.add(current_vertex)
        for neighbor, weight in graph_adjacency[current_vertex]:
            distance = Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight)))
            if distance<distances[neighbor]:
                distances[neighbor] = Weight(tuple(x + y for x, y in zip(current_distance.weight,weight.weight)))
                heapq.heappush(pq, (distance, neighbor))
    return distances
graph_dict=[
    {'start':'a','end':'b','weight':(1,10,45)},
    {'start':'b','end':'c','weight':(2,5,55)},
    {'start':'c','end':'d','weight':(5,5,23)},
    {'start':'a','end':'d','weight':(8,1,66)}]

print(*list(i for i in compute_distances('a','c',graph_dict).values()))