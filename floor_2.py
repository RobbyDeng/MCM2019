import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

#reserve the escape path of each block
node_list = []

#popularity of each block
p= np.array([1,1,1,1,1,1,1,2,1,1,2,1,1,1,
             1,1,1,1,2,1,2,2,1,1,2,1,1,0,0,
             0,0,0,0,0])   
#area of each block
S=495 
s= np.array([S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S
            ,S,S,S,S,S,S,S,0,0,0,0,0,0,0]) 

n0 = 27413       # total number of tourists
a=50   #customized parameter in model                                          
b=20   #customized parameter in model
e=2400 #common escaping efficiency (individuals/minute)

s_sum = sum(s)
PS = p*s
p_sum = sum(PS) #sum up the popularity
p_avg = p_sum/s_sum
print("p_sum=",p_sum)

#popularity
popu1=1
popu2=2

#reserve escaping time of each block
TIME_Array = [] 

def quadratic_equation(a, b, c):  
    t = math.sqrt(pow(b, 2) - 4 * a * c)  
    if(pow(b, 2) - 4 * a * c) > 0:  
        return (-b + t) / (2 * a), (-b - t) / (2 * a)     
    elif (pow(b, 2) - 4 * a * c) == 0:   
        return (-b + t) / (2 * a)  
    else:  
        return None  

def Dijkstra(G,start,end):
    RG = G.reverse(); dist = {}; previous = {}
    for v in RG.nodes():
        dist[v] = float('inf')
        previous[v] = 'none'
    dist[end] = 0
    u = end
    while u!=start:
        u = min(dist, key=dist.get)
        distu = dist[u]
        del dist[u]
        for u,v in RG.edges(u):
            if v in dist:
                alt = distu + RG[u][v]['weight']
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
    path=(start,)
    last= start
    while last != end:
        nxt = previous[last]
        path += (nxt,)
        last = nxt
    return path

L= np.array([30,22,22,22,30,30,30,8,0,30,22,22,38,30,
             22,30,30,38,0,0,0,22,22,30,0,8,0,0]) # route length
PC=np.array([1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,
             2,2,1,1,1,2,1]) #reflect popularity
LC=(L*PC)/p_avg              #"optical path"
print(LC)

G=nx.DiGraph()

G.add_edge(0,1,weight=LC[0])
G.add_edge(1,4,weight=LC[1])
G.add_edge(2,1,weight=LC[2])
G.add_edge(3,5,weight=LC[3])
G.add_edge(5,8,weight=LC[4])
G.add_edge(8,7,weight=LC[5])
G.add_edge(7,6,weight=LC[6])
G.add_edge(6,27,weight=LC[7])
G.add_edge(4,27,weight=LC[8])
G.add_edge(9,6,weight=LC[9])
G.add_edge(10,9,weight=LC[10])
G.add_edge(12,11,weight=LC[11])
G.add_edge(11,15,weight=LC[12])
G.add_edge(15,16,weight=LC[13])
G.add_edge(13,14,weight=LC[14])
G.add_edge(14,15,weight=LC[15])
G.add_edge(16,20,weight=LC[16])
G.add_edge(16,17,weight=LC[17])
G.add_edge(20,30,weight=LC[18])
G.add_edge(17,28,weight=LC[19])
G.add_edge(18,29,weight=LC[20])
G.add_edge(19,18,weight=LC[21])
G.add_edge(21,19,weight=LC[22])
G.add_edge(23,26,weight=LC[23])
G.add_edge(26,33,weight=LC[24])
G.add_edge(25,33,weight=LC[25])
G.add_edge(24,32,weight=LC[26])
G.add_edge(22,31,weight=LC[27])


Dij_path_11_28 = Dijkstra(G,11,28)
Dij_path_11_30 = Dijkstra(G,11,30)

Dij_path_12_28 = Dijkstra(G,12,28)
Dij_path_12_30 = Dijkstra(G,12,30)

Dij_path_13_28 = Dijkstra(G,13,28)
Dij_path_13_30 = Dijkstra(G,13,30)

Dij_path_14_28 = Dijkstra(G,14,28)
Dij_path_14_30 = Dijkstra(G,14,30)

Dij_path_15_28 = Dijkstra(G,15,28)
Dij_path_15_30 = Dijkstra(G,15,30)

Dij_path_16_28 = Dijkstra(G,16,28)
Dij_path_16_30 = Dijkstra(G,16,30)

print(Dij_path_13_28)
print(Dij_path_13_30)


#Node_1
A0 = (-1)*a*e*popu1*0.5/p_sum   
B0 = b+a*n0*popu1/p_sum         
C0 = LC[1]*(-1)             
t1_4 = abs(max(quadratic_equation(A0, B0, C0)))

A1=(-1)*a*e*popu1*0.5/p_sum
B1=b+a*n0*popu1/p_sum-a*e*popu1*t1_4*0.5/p_sum 
C1=LC[8]*(-1)               
t4_27 = abs(max(quadratic_equation(A1, B1, C1)))

t1_27=t1_4+t4_27
node_list.append(t1_27)
print("t1_27=",t1_27)

#Node_2

A0 = (-1)*a*e*popu1*0.5/p_sum   
B0 = b+a*n0*popu1/p_sum         
C0 = LC[2]*(-1)             
t2_1 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*popu1*0.5/p_sum
B1=b+a*n0*popu1/p_sum-a*e*popu1*t2_1/p_sum
C1=LC[1]*(-1)
t1_4 = abs(max(quadratic_equation(A1, B1, C1)))

A2=(-1)*a*e*popu1*0.5/p_sum
B2=b+a*n0*popu1/p_sum-a*e*popu1*(t2_1+t1_4)/p_sum
C2=LC[8]*(-1)
t4_27 = abs(max(quadratic_equation(A2, B2, C2)))

t2_27 = t2_1+t1_4+t4_27
node_list.append(t2_27)

print("t2_27=",t2_27)


#Node_3 一元二次方程无解

A0 = (-1)*a*e*popu1*0.5/p_sum   
B0 = b+a*n0*popu1/p_sum         
C0 = LC[3]*(-1)          
t3_5 = abs(max(quadratic_equation(A0, B0, C0)))

A1=(-1)*a*e*popu1*0.5/p_sum
B1=b+a*(n0-e*t3_5)*popu1/p_sum
C1=LC[4]*(-1)
t5_8 = abs(max(quadratic_equation(A1, B1, C1)))

A2=(-1)*a*e*popu1*0.5/p_sum
B2=b+a*(n0-e*(t3_5+t5_8))*popu1/p_sum
C2=LC[5]*(-1)
t8_7 = abs(max(quadratic_equation(A2, B2, C2)))

A3=(-1)*a*e*popu1*0.5/p_sum
B3=b+a*(n0-e*(t3_5+t5_8+t8_7))*popu1/p_sum
C3=LC[6]*(-1)
t7_6 = abs(max(quadratic_equation(A3, B3, C3)))

A4=(-1)*a*e*popu1*0.5/p_sum
B4=b+a*(n0-e*(t3_5+t5_8+t8_7+t7_6))*popu1/p_sum
C4=LC[7]*(-1)
t6_27 = abs(max(quadratic_equation(A4, B4, C4)))

t3_27 = t3_5+t5_8+t8_7+t7_6+t6_27
node_list.append(t3_27)
print("t3_27=",t3_27)


#Node#Node_16_28

A3=(-1)*a*e*popu1*0.5/p_sum
B3=b+a*n0*popu1/p_sum
C3=LC[17]*(-1)
t16_17 = max(quadratic_equation(A3, B3, C3))

A4=(-1)*a*e*popu1*0.5/p_sum
B4=b+a*n0*popu1/p_sum-a*e*popu1*(t16_17)*0.5/p_sum
C4=LC[19]*(-1)
t17_28 = abs(max(quadratic_equation(A4, B4, C4)))

t16_28 = t16_17+t17_28
node_list.append(t16_28)
print("t16_28=",t16_28)

#Node_16_30

A3=(-1)*a*e*popu1*0.5/p_sum
B3=b+a*n0*popu1/p_sum
C3=LC[16]*(-1)
t16_20 = max(quadratic_equation(A3, B3, C3))

A4=(-1)*a*e*popu1*0.5/p_sum
B4=b+a*n0*popu1/p_sum-a*e*popu1*(t16_20)*0.5/p_sum
C4=LC[18]*(-1)
t20_30 = abs(max(quadratic_equation(A4, B4, C4)))

t16_30 = t16_20+t20_30
node_list.append(t16_30)
print("t16_30=",t16_30)


plt.bar(range(len(node_list)), node_list, color = 'lightsteelblue')

plt.xticks(range(len(node_list)), node_list)
plt.xlabel('node')
plt.ylabel("escaping time")
plt.legend()
plt.show()



