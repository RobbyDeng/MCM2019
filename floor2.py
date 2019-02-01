import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math


p= np.array([1,1,1,1,1,1,1,2,1,1,2,1,1,1,
             1,1,1,1,2,1,2,2,1,1,2,1,1,0,0,
             0,0,0,0,0])    #受欢迎程度
S=495 #每一个块的面积
s= np.array([S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S
            ,S,S,S,S,S,S,S,0,0,0,0,0,0,0])    #面积

n0 = 27413       #馆内总人数
a=2
b=0   #参数用于决定人群移动速度
e=300 #单位时间逃离的人数

s_sum = sum(s)
PS = p*s
p_sum = sum(PS) #受欢迎程度求和
p_avg = p_sum/s_sum


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
             22,30,30,38,0,0,0,22,22,30,0,8,0,0]) #按序排列的边的长度
PC=np.array([1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,
             2,2,1,1,1,2,1]) #想当于n 就是起点的受欢迎程度
LC=(L*PC)/p_avg              #按序排列的边的光程
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

'''
print(Dij_path_13_28)
print(Dij_path_13_30)
'''

'''
#Node_0
A0 = (-1)*a*e*1*0.5/p_sum
B0 = b+a*n0*1/p_sum
C0 = LC[0]*(-1)
t0_1 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t0_1*0.5/p_sum
C1=LC[1]*(-1)
t1_4=max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t0_1+t1_4)*0.5/p_sum
C2=LC[8]*(-1)
t4_27 = max(quadratic_equation(A2, B2, C2))

t0 = t0_1+t1_4+t4_27
print(t0)
'''

#Node_1
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[1]*(-1)             #对应LC中哪条路径
t1_4 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t1_4*0.5/p_sum  #改时间下标
C1=LC[8]*(-1)               #改路径
t4_27 = max(quadratic_equation(A1, B1, C1))

t1=t1_4+t4_27
print(t1)

#Node_2
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[2]*(-1)             #对应LC中哪条路径
t2_1 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t2_1*0.5/p_sum
C1=LC[1]*(-1)
t1_4 = max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t2_1+t1_4)*0.5/p_sum
C2=LC[8]*(-1)
t4_27 = max(quadratic_equation(A2, B2, C2))

t2 = t2_1+t1_4+t4_27
print(t2)

#Node_3 一元二次方程无解
'''
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[3]*(-1)             #对应LC中哪条路径
t3_5 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*(n0-e*t3_5)*1/p_sum
C1=LC[4]*(-1)
t5_8 = max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*(n0-e*(t3_5+t5_8))*1/p_sum
C2=LC[5]*(-1)
t8_7 = max(quadratic_equation(A2, B2, C2))

A3=(-1)*a*e*1*0.5/p_sum
B3=b+a*(n0-e*(t3_5+t5_8+t8_7))*1/p_sum
C3=LC[6]*(-1)
t7_6 = max(quadratic_equation(A3, B3, C3))

A4=(-1)*a*e*1*0.5/p_sum
B4=b+a*(n0-e*(t3_5+t5_8+t8_7+t7_6))*1/p_sum
C4=LC[7]*(-1)
t6_27 = max(quadratic_equation(A4, B4, C4))

t3 = t3_5+t5_8+t8_7+t7_6+t6_27
print(t3)
'''
#Node_4
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[8]*(-1)             #对应LC中哪条路径
t4_27 = max(quadratic_equation(A0, B0, C0))

t4 = t4_27
print(t4)

#Node_5     一元二次方程无解
'''
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[4]*(-1)             #对应LC中哪条路径
t5_8 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*(n0-e*t5_8)*1/p_sum
C1=LC[5]*(-1)
t8_7 = max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*(n0-e*(t5_8+t8_7))*1/p_sum
C2=LC[6]*(-1)
t7_6 = max(quadratic_equation(A2, B2, C2))

A3=(-1)*a*e*1*0.5/p_sum
B3=b+a*(n0-e*(t5_8+t8_7+t7_6))*1/p_sum
C3=LC[7]*(-1)
t6_27 = max(quadratic_equation(A3, B3, C3))

t5 = t5_8+t8_7+t7_6+t6_27
print(t5)
'''

#Node_6
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[7]*(-1)             #对应LC中哪条路径
t6_27 = max(quadratic_equation(A0, B0, C0))

t6 = t6_27
print(t6)

#Node_7
A0 = (-1)*a*e*2*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[6]*(-1)             #对应LC中哪条路径
t7_6 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t7_6*0.5/p_sum
C1=LC[7]*(-1)
t6_27 = max(quadratic_equation(A1, B1, C1))

t7 = t7_6+t6_27
print(t7)

#Node_7
A0 = (-1)*a*e*2*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[6]*(-1)             #对应LC中哪条路径
t7_6 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t7_6*0.5/p_sum
C1=LC[7]*(-1)
t6_27 = max(quadratic_equation(A1, B1, C1))

t7 = t7_6+t6_27
print(t7)

#Node_8
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[5]*(-1)             #对应LC中哪条路径
t8_7 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t8_7*0.5/p_sum
C1=LC[6]*(-1)
t7_6 = max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t8_7+t7_6)*0.5/p_sum
C2=LC[7]*(-1)
t6_27 = max(quadratic_equation(A2, B2, C2))

t8 = t8_7+t7_6+t6_27
print(t8)

#Node_9
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[9]*(-1)             #对应LC中哪条路径
t9_6 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t9_6*0.5/p_sum
C1=LC[7]*(-1)
t6_27 = max(quadratic_equation(A1, B1, C1))

t8 = t9_6+t6_27
print(t9)

#Node_10
A0 = (-1)*a*e*2*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*2/p_sum         #本节点的受欢迎程度
C0 = LC[10]*(-1)             #对应LC中哪条路径
t10_9 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t10_9*0.5/p_sum
C1=LC[9]*(-1)
t9_6= max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t10_9+t9_6)*0.5/p_sum
C2=LC[7]*(-1)
t6_27 = max(quadratic_equation(A2, B2, C2))

t10 = t10_9+t9_6+t6_27
print(t10)

#Node_11_28
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[12]*(-1)             #对应LC中哪条路径
t11_15 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t11_15*0.5/p_sum
C1=LC[13]*(-1)
t15_16= max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t11_15+t15_16)*0.5/p_sum
C2=LC[17]*(-1)
t16_17 = max(quadratic_equation(A2, B2, C2))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t11_15+t15_16+t16_17)*0.5/p_sum
C2=LC[19]*(-1)
t17_28 = max(quadratic_equation(A2, B2, C2))

t11_28 = t11_15+t15_16+t16_17+t17_28
print(t11_28)

#Node_11_30
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[12]*(-1)             #对应LC中哪条路径
t11_15 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t11_15*0.5/p_sum
C1=LC[13]*(-1)
t15_16= max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t11_15+t15_16)*0.5/p_sum
C2=LC[16]*(-1)
t16_20 = max(quadratic_equation(A2, B2, C2))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t11_15+t15_16+t16_20)*0.5/p_sum
C2=LC[18]*(-1)
t20_30 = max(quadratic_equation(A3, B3, C3))

t11_30 = t11_15+t15_16+t16_20+t20_30
print(t11_30)

#Node_12_28
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[11]*(-1)             #对应LC中哪条路径
t12_11 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t12_11*0.5/p_sum
C1=LC[12]*(-1)
t11_15= max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15)*0.5/p_sum
C2=LC[13]*(-1)
t15_16 = max(quadratic_equation(A2, B2, C2))

A3=(-1)*a*e*1*0.5/p_sum
B3=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15+t15_16)*0.5/p_sum
C3=LC[17]*(-1)
t16_17 = max(quadratic_equation(A3, B3, C3))

A4=(-1)*a*e*1*0.5/p_sum
B4=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15+t15_16+t16_17)*0.5/p_sum
C4=LC[19]*(-1)
t17_28 = max(quadratic_equation(A4, B4, C4))

t12_28 = t12_11+t11_15+t15_16+t16_17+t17_28
print(t12_28)

#Node_12_30
A0 = (-1)*a*e*1*0.5/p_sum   #本节点的受欢迎程度
B0 = b+a*n0*1/p_sum         #本节点的受欢迎程度
C0 = LC[11]*(-1)             #对应LC中哪条路径
t12_11 = max(quadratic_equation(A0, B0, C0))

A1=(-1)*a*e*1*0.5/p_sum
B1=b+a*n0*1/p_sum-a*e*1*t12_11*0.5/p_sum
C1=LC[12]*(-1)
t11_15= max(quadratic_equation(A1, B1, C1))

A2=(-1)*a*e*1*0.5/p_sum
B2=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15)*0.5/p_sum
C2=LC[13]*(-1)
t15_16 = max(quadratic_equation(A2, B2, C2))

A3=(-1)*a*e*1*0.5/p_sum
B3=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15+t15_16)*0.5/p_sum
C3=LC[17]*(-1)
t16_17 = max(quadratic_equation(A3, B3, C3))

A4=(-1)*a*e*1*0.5/p_sum
B4=b+a*n0*1/p_sum-a*e*1*(t12_11+t11_15+t15_16+t16_17)*0.5/p_sum
C4=LC[19]*(-1)
t17_28 = max(quadratic_equation(A4, B4, C4))

t12_28 = t12_11+t11_15+t15_16+t16_17+t17_28
print(t12_28)



#有一个问题：到这里为止输出的结果不太符合实际情况，按理说Node6离出口较近，花的时间较短才对，还有之前的一些点花的时间。。。


#print(rs)

#绘制网络图G，带标签
#nx.draw(G,with_labels = True,node_color='r')
#plt.show()

