import torch 
import numpy as np 

graph_matrix=[[[0,1],[1,2],[0,1]],[[0],[1],[2]],[[0],[1,2],[2]]]
connections = np.zeros([12,12],dtype = int)
#print(connections[0,0])
#print(connections)

def network_tom(graph_matrix, connection ):
    for ind, k in enumerate(graph_matrix):
        for i, m in enumerate(k):
            for item in m:
                connection[item+3*ind, i+3*ind+3]=1
    return connection

#connections = network_tom(graph_matrix, connections)
#print(connections)

def network_tom2(graph_matrix, connection ):
    for ind, k in enumerate(graph_matrix):
        for i, m in enumerate(k):
            for item in m:
                connection[item+3*ind, i+3*ind+6]=1
    return connection


def network_tom3(graph_matrix, connection):
    for ind, k in enumerate(graph_matrix):
        for i, m in enumerate(k):
            for item in m:
                connection[item+3*ind, i+3*ind+9]=1
    return connection

def mto_network(connections):
    graph=[]
    for i in range(3):
        sub_graph = connections[i*3:i*3+3,i*3+3:i*3+6] 
        #print(sub_graph)
        temp2 =[]
        for i in range(3):
            temp = []
            for index, item in enumerate(sub_graph[:,i]):
                if item ==1:
                    temp.append(index)
            temp2.append(temp)
        graph.append(temp2)
        
    return graph

def mto_2(connections):
    graph = []
    for i in range(2):
        sub_graph = connections[i*3:i*3+3,i*3+6:i*3+9]  
        temp2 =[]
        for i in range(3):
            temp = []
            for index, item in enumerate(sub_graph[:,i]):
                if item ==1:
                    temp.append(index)
            temp2.append(temp)
        graph.append(temp2)  
    return graph

#print(mto_2(connections))


def mto_3(connections):
    graph = []
    for i in range(1):
        sub_graph = connections[i*3:i*3+3,i*3+9:i*3+12]  
        temp2 =[]
        for i in range(3):
            temp = []
            for index, item in enumerate(sub_graph[:,i]):
                if item ==1:
                    temp.append(index)
            temp2.append(temp)
        graph.append(temp2)  
    return graph 

#print(mto_3(connections))
"""
def test_mto2(connections):
    mto_2(connections)
"""
def sec_rand(connection):
    i = np.random.randint(low=0, high=2, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+6:i*3+9]
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0
    connections[i*3:i*3+3,i*3+6:i*3+9] = sub_graph
    return connection

def third_rand(connection):
    i=0
    sub_graph = connection[i*3:i*3+3,i*3+9:i*3+12]
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0
    connection[i*3:i*3+3,i*3+9:i*3+12] = sub_graph
    
    return connection



    
"""
def rand_gene(connection):
    for i in range(3):
        sub_graph = connection[i*3:i*3+3,i*3+3:i*3+6] 
        inx=np.random.randint(low=0, high=3, size=1, dtype='l')
        iny=np.random.randint(low=0, high=3, size=1, dtype='l')
        #print(inx,iny)
        sub_graph[inx,iny] = 1
        connection[i*3:i*3+3,i*3+3:i*3+6] = sub_graph
    return connection
"""
def rand_gene(connection):
    i = np.random.randint(low=0, high=3, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+3:i*3+6] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    sub_graph[inx,iny] = 1
    connection[i*3:i*3+3,i*3+3:i*3+6] = sub_graph
    i = np.random.randint(low=0, high=3, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+3:i*3+6] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0
    
    for tt in range(3):
        if sum(sub_graph[:,tt]) ==0:
            sub_graph[tt,tt] =1

    connection[i*3:i*3+3,i*3+3:i*3+6] = sub_graph
    #for i in range(9):
        #connection[i,i+3]=1
    return connection

def newrand(con):
    connection = con
    """
    i = np.random.randint(low=0, high=3, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+3:i*3+6] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    sub_graph[inx,iny] = 1
    connection[i*3:i*3+3,i*3+3:i*3+6] = sub_graph

    i = np.random.randint(low=0, high=2, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+6:i*3+9] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    sub_graph[inx,iny] = 1
    connection[i*3:i*3+3,i*3+6:i*3+9] = sub_graph
    """
    i = np.random.randint(low=0, high=3, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+3:i*3+6] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0

    for tt in range(3):
        if sum(sub_graph[:,tt]) ==0:
            sub_graph[tt,tt] =1
    connection[i*3:i*3+3,i*3+3:i*3+6] = sub_graph

    i = np.random.randint(low=0, high=2, size=1, dtype='l')
    i = int(i)
    sub_graph = connection[i*3:i*3+3,i*3+6:i*3+9] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0
    """
    for tt in range(3):
        if sum(sub_graph[:,tt]) ==0:
            sub_graph[tt,tt] =1
    """
    connection[i*3:i*3+3,i*3+6:i*3+9] = sub_graph

    i = int(0)
    sub_graph = connection[i*3:i*3+3,i*3+9:i*3+12] 
    inx=np.random.randint(low=0, high=3, size=1, dtype='l')
    iny=np.random.randint(low=0, high=3, size=1, dtype='l')
    #print(inx,iny)
    if sub_graph[inx,iny] == 0:
        sub_graph[inx,iny] = 1
    else:
        sub_graph[inx,iny] = 0
    """
    for tt in range(3):
        if sum(sub_graph[:,tt]) ==0:
            sub_graph[tt,tt] =1
    """
    connection[i*3:i*3+3,i*3+9:i*3+12] = sub_graph

    #for i in range(9):
        #connection[i,i+3]=1
        
    return connection

#print(rand_gene(connections))
#def new_generation(connection1, connection2):
    

#print(mto_network(connections))


class population():
    def __init__(self, graph_matrix = [[[0],[1],[2]],[[0],[1],[2]],[[0],[1],[2]]], graph2 = [[[],[],[]],[[],[],[]]], graph3 = [[[],[],[]]]):
        self.graph_matrix = graph_matrix
        self.graph2 = graph2
        self.graph3 = graph3
        self.connections = np.zeros([12,12],dtype = int)
        self.connections = network_tom(self.graph_matrix,self.connections)
        self.connections = network_tom2(self.graph2, self.connections)
        self.connections = network_tom3(self.graph3, self.connections)
    def create_radom(self):
        self.connections = network_tom(self.graph_matrix,self.connections)
        self.connections = rand_gene(self.connections)
        self.graph_matrix = mto_network(self.connections)
    def create_deeprand(self):
        self.connections = network_tom(self.graph_matrix,self.connections)
        self.connections = network_tom2(self.graph2, self.connections)
        self.connections = network_tom3(self.graph3, self.connections)
        self.connections = newrand(self.connections)
        self.graph_matrix = mto_network(self.connections)
        self.graph2 = mto_2(self.connections)
        self.graph3 = mto_3(self.connections)
        
    def generate(self, connect):
        connect = newrand(connect)
        graph_matrix = mto_network(connect)
        graph2 = mto_2(connect)
        graph3 = mto_3(connect)
        return connect, graph_matrix, graph2, graph3
    
    def status(self):
        self.graph_matrix = mto_network(self.connections)
        self.graph2 = mto_2(self.connections)
        self.graph3 = mto_3(self.connections)


def test_pop():
    p1 = population()
    p2 = population()
    print('p1', p1.graph_matrix)
    print('p2', p2.graph_matrix)
    for i in range(100):
        p1.create_radom()
        p2.create_radom()
        print('p1', p1.graph_matrix)
        print('p1', p1.connections)
        print('p2', p2.graph_matrix)

def test_pop_deep():
    p1 = population()
    p2 = population()
    print('p1', p1.connections)
    print('p2', p2.connections)
    """
    for i in range(100):
        p1.create_deeprand()
        p2.create_deeprand()
        print('p1', p1.connections)
        #print('p2', p2.connections)
        print('p1', p1.graph_matrix)
        print('p1', p1.graph2)
        print('p1', p1.graph3)
    """
    for i in range(100):
        p2.connections,p2.graph_matrix,p2.graph2,p2.graph3 = p2.generate(p1.connections)
        print('p1', p1.connections)
        print('p2', p2.connections)
        density = sum(p2.connections)/54
        print('density',density)
        print('p1', p1.graph_matrix)
        print('p1', p1.graph2)
        print('p1', p1.graph3)
        print('p2', p2.graph_matrix)
        print('p2', p2.graph2)
        print('p2', p2.graph3)

#test_pop()
#test_pop_deep()
def test_x():
    p1 = population()
    print(p1.connections)
    print(p1.status())

#test_x()