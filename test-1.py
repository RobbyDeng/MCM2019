import numpy as np
import matplotlib as plot

from collections import defaultdict
from heapq import *

p= np.array([1,1,2,2,3,3,0,0])    #受欢迎程度
s= np.array([12,15,10,9,8,11,0,0])#面积
n0 = 10000                      #馆内总人数
c0=0.9 #初始速度
a=1
b=0   #参数用于决定人群移动速度
e=300 #单位时间逃离的人数
q0=3  #分界人流密度
w=100000 #用于表示距离无穷

'''
D=np.mat([[0,w,w,w,3,w,w,w],
         [w,0,w,w,w,1,w,w],
         [w,w,0,w,2,w,w,w],
         [w,w,w,0,w,3,w,w],
         [3,w,2,w,0,5,7,w],
         [w,1,w,3,5,0,w,8],
         [w,w,w,w,7,w,0,w],
         [w,w,w,w,w,8,w,0]]
)
'''

s_sum = sum(s)
p_sum = sum(p*s) #受欢迎程度求和
p_avg = p_sum/s_sum
print(p_avg)
#N = n0 - e*t

#########


def dijkstra_raw(edges, from_node, to_node):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))
    q, seen = [(0,from_node,())], set()
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == to_node:
                return cost,path
            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost+c, v2, path))
    return float("inf"),[]

def dijkstra(edges, from_node, to_node):
    len_shortest_path = -1
    ret_path=[]
    length,path_queue = dijkstra_raw(edges, from_node, to_node)
    if len(path_queue)>0:
        len_shortest_path = length        ## 1. Get the length firstly;
        ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
        left = path_queue[0]
        ret_path.append(left)        ## 2.1 Record the destination node firstly;
        right = path_queue[1]
        while len(right)>0:
            left = right[0]
            ret_path.append(left)    ## 2.2 Record other nodes, till the source-node.
            right = right[1]
        ret_path.reverse()    ## 3. Reverse the list finally, to make it be normal sequence.
    return len_shortest_path,ret_path

### ==================== Given a list of nodes in the topology shown in Fig. 1.
list_nodes_id = [0,1,2,3,4,5,6,7];
### ==================== Given constants matrix of topology.
### M_topo is the 2-dimensional adjacent matrix used to represent a topology.
M_topo = [[[w,w,w,w,1,w,w,w],
           [w,w,w,w,w,1,w,w],
           [w,w,w,w,1,w,w,w],
           [w,w,w,w,w,1,w,w],
           [1,w,1,w,w,1,1,w],
           [w,1,w,1,1,w,w,1],
           [w,w,w,w,1,w,w,w],
           [w,w,w,w,w,1,w,w]]
          ]

### --- Read the topology, and generate all edges in the given topology.
edges = []
for i in range(len(M_topo)):
    for j in range(len(M_topo[0])):
        if i!=j and M_topo[i][j]!=w:
            edges.append((i,j,M_topo[i][j]))### (i,j) is a link; M_topo[i][j] here is 1, the length of link (i,j).
print ("=== Dijkstra ===")
print ("Let's find the shortest-path from 0 to 6:")
length,Shortest_path = dijkstra(edges, 4, 6)
print ('length = ',length)
print ('The shortest path is ',Shortest_path)











