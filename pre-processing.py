import networkx as nx
import pickle as pk
import os,glob
import matplotlib.pyplot as plt 
import seaborn as sns
from expts.gendata import gen_sbm
from expts.gengraph import random_sbm

#code for graph analysis
def analyze_graph():
    data_path = 'data/rand_sbm'
    for filename in glob.glob(os.path.join(data_path, '*.pkl')):
        with open(filename, 'rb') as f:
            graph = pk.load(f)
            sort_components = sorted(nx.connected_components(graph), key = len, reverse=True)
            average_degree = sum([d for (n, d) in graph.degree]) / float(len(graph.nodes))
            print(filename, ' & ', len(graph.nodes), ' & ', len(graph.edges), ' & ', average_degree, '\\\\')
            #for component in sort_components:
                #print(len(sort_components))
            f.close()
    node_number = []
    for i in range(10):
        with open(data_path+'/rand_sbm_'+str(i)+'.pkl', 'rb') as f:
            graph = pk.load(f)
            node_number.append(len(graph.nodes))
            f.close()
    print(node_number)

def visualize_graph():
    #data_path = "data/mammal/rob.pkl"
    data_path = 'data/rt/copen_occupy.pkl'
    #data_path = 'data/rt/damascus.pkl'
    with open(data_path,'rb') as f:
        graph = pk.load(f)
        sort_components = sorted(nx.connected_components(graph), key = len, reverse=True)
        nx.draw(graph, with_labels=False, node_size=10, node_color="blue", node_shape="o", alpha=0.5, linewidths=2, font_size=25, font_color="grey", font_weight="bold", width=1, edge_color="black")
        plt.savefig('data/mammal/rob.png')
        print(len(graph.nodes), len(graph.edges))
        f.close()


#code to combine graphs
def combine_graph():
    #data_path1 = 'data/rt/voteonedirection.pkl'
    #data_path2 = 'data/rt/copen.pkl'
    #data_path1 = 'data/rand_data/rand_100.pkl'
    #data_path2 = 'data/rand_data/rand_500.pkl'
    #data_path1 = 'data/rt/voteonedirection.pkl'
    #data_path2 = 'data/rt/israel.pkl'
    num_graphs = 5
    for i in range(num_graphs):
        #j = i #for 0-4
        j = i+5 #for 5-9
        k=j+1 if i != num_graphs-1 else j-num_graphs+1 
        data_path1 = 'data/rand_sbm/rand_sbm_'+str(j)+'.pkl'
        data_path2 = 'data/rand_sbm/rand_sbm_'+str(k)+'.pkl'
        new_data_path = 'data/rand_sbm/rand_sbm_'+str(j)+'_'+str(k)+'.pkl'

        with open(data_path1,'rb') as f:
            graph1 = pk.load(f)
            f.close()
        with open(data_path2,'rb') as f:
            graph2 = pk.load(f)
            f.close()

        graph = nx.disjoint_union(graph1, graph2)
        print(len(graph))
        with open(new_data_path,'wb') as f:
            pk.dump(graph, f)
            f.close()

def generate_graph(num=10,n=500, c=5, p_w=0.02, p_b=0.005, std=20, output_file="data/rand_sbm/rand_sbm"):

    for i in range(num):
        graph = random_sbm(n,c,p_w,p_b,std)
        data_path = output_file+'_'+str(i)+'.pkl'
        with open(data_path, 'wb') as f:
            pk.dump(graph,f)
            f.close()
    #gen_sbm(num=5,n=1000, c=10, p_w=0.1, p_b=0.01, std=30, output_file="rand_data/rand_sbm_5_10000_10.pkl")


#generate_graph()
analyze_graph()
#combine_graph()
