from collections import defaultdict
from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from random import randrange
import statistics
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def CountTrianglesColor(edges):
    # Create a defaultdict to store the neighbors of each vertex    
    neighbors = defaultdict(set)
    if (edges[0] == -1):
        return [0]
    else:
        for edge in edges[1]:
            u, v = edge           
                 
            neighbors[u].add(v)
            neighbors[v].add(u)
    
        # Initialize the triangle count to zero
        triangle_count = 0
    
        # Iterate over each vertex in the graph.
        # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
        for u in neighbors:
            # Iterate over each pair of neighbors of u
            for v in neighbors[u]:            
                if v > u:
                    for w in neighbors[v]:
                        # If w is also a neighbor of u, then we have a triangle
                        if w > v and w in neighbors[u]:
                            triangle_count += 1
        # Return the total number of triangles in the graph        
        return [triangle_count]
    
    
def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity    
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return [triangle_count]

def MR_ExactTC(edges, C):
    start_time = int(round(time.time() * 1000000))
    p = 8191
    a = (rand.randint(0, p-1))
    b = (rand.randint(0, p-1))
    
    triangles = (edges.flatMap(lambda edge: GenerateKeys(edge, C, a, b, p))
                 .groupByKey()
                 .flatMap(lambda edge: countTriangles2(edge[0], edge[1], a, b, p, C))
                 .sum()
                 )

    elapsed_time = (int(round(time.time() * 1000000)) - start_time) // 1000000
    return triangles, elapsed_time

def GenerateKeys(edge, C, a, b, p):
    def hashColor(vert):
        return ((a*vert + b) % p) % C
    u, v = edge
    generatedPairs = []
    coloredEdges = [hashColor(u), hashColor(v)]
    for i in range(C):
        temp = coloredEdges.copy()
        temp.append(i)
        generatedPairs.append((tuple(sorted(temp)), edge))

    return generatedPairs
    

def MR_ApproxTCwithNodeColors(edges, C):
    start_time = int(round(time.time() * 1000000))
    p = 8191
    a = (rand.randint(0, p-1))
    b = (rand.randint(0, p-1))
    def hashColor(vert):
        return ((a*vert + b) % p) % C
    
    triangles = (edges.flatMap(lambda edge: [(hashColor(edge[0]), edge)] 
                                   if (hashColor(edge[0]) == hashColor(edge[1])) 
                                   else [(-1,(-2,-2))])                   
                     .groupByKey()
                     .flatMap(CountTrianglesColor)
                     .sum()*(C**2)  
                     )

    elapsed_time = (int(round(time.time() * 1000000)) - start_time) // 1000000
    return triangles, elapsed_time

def main():
    if len(sys.argv) != 5:
        print("Usage: python G099HW2.py <C> <R> <F> <file_path>")
        sys.exit(1)
    global C, R, F
    C, R, F, data_path = sys.argv[1:]
    if not C.isdigit() or not R.isdigit():
        raise ValueError("C and R must be both integers")
    if int(F) not in [0, 1]:
        raise ValueError("F must be either 0 or 1")
    C, R, F = int(C), int(R), int(F)
    conf = SparkConf().set("spark.ui.showConsoleProgress", "false").setAppName('G099HW2')
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext.getOrCreate(conf=conf)
    if not sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration()).exists(sc._gateway.jvm.org.apache.hadoop.fs.Path(data_path)):
        raise FileNotFoundError("File not found: " + data_path)
        
    input = sc.textFile(data_path).repartition(32).sortBy(lambda x: randrange(1000000)).cache()
    edges = input.map(lambda x: tuple(map(int, x.strip().split(',')))).repartition(32)
    numedges = edges.count()    
    estimates = []
    running_times = []
    if F == 0:
        for i in range(R):
            estimate, run_time = MR_ApproxTCwithNodeColors(edges, C)
            estimates.append(estimate)
            running_times.append(run_time)
        median_estimate = round(statistics.median(estimates))
        median_time = round(sum(running_times) / R)
        # Print outputs
        print("Dataset = ", data_path)
        print("Number of Edges = ", numedges)
        print('Number of Colors = ', C)
        print('Number of Repetitions  = ', R)
        print('Approximation algorithm with node coloring')
        print('- Number of triangles (median over ', R, ' runs) = ', median_estimate)
        print("- Running time (average over ", R, " runs) = ", median_time, 'ms') 
        
    else:
        for i in range(R):
            nTriang, run_time = MR_ExactTC(edges, C)            
            running_times.append(run_time)               
        median_time = round(sum(running_times) / R)        
        # Print outputs
        print("Dataset = ", data_path)
        print("Number of Edges = ", numedges)
        print('Number of Colors = ', C)
        print('Number of Repetitions  = ', R)
        print('Exact algorithm with node coloring')
        print('- Number of triangles = ', nTriang)
        print("- Running time (average over ", R, " runs) = ", median_time, 'ms')   

    

if __name__ == "__main__":
    main()
