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
    return triangle_count

def MR_ExactTC(edges, C):
    p = 8191
    a = (rand.randint(0, p-1))
    b = (rand.randint(0, p-1))
    def hashColor(vert):
        return ((a*vert + b) % p) % C
    triangles = (edges.flatMap(lambda edge: GenerateKeys(edge, C, a, b, p))
                 .groupByKey()
                 )
    return triangles

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
    start_time = time.time_ns()
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
                     .sum()*(C**2)  )

    elapsed_time = (time.time_ns() - start_time) // 1000000
    return triangles, elapsed_time

def main():
    assert len(sys.argv) == 4, "Usage: python G099HW1.py <C> <R> <file_path>"
    global C, R  
    C, R, data_path = sys.argv[1:]
    assert C.isdigit() and R.isdigit(), "K and H must be both integers"
    C, R = int(C), int(R)
    assert os.path.isfile(data_path), "File or folder not found"
    conf = SparkConf().set("spark.ui.showConsoleProgress", "false").setAppName('G099HW1').setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)

    input = sc.textFile(data_path).sortBy(lambda x: randrange(1000000))
    edges = input.map(lambda x: tuple(map(int, x.strip().split(','))))
    numedges = edges.count()
    smth = MR_ExactTC(edges, C).collect()
    print(smth)
"""
    estimates = []
    running_times = []
    for i in range(R):
        estimate, run_time = MR_ApproxTCwithNodeColors(edges, C)
        estimates.append(estimate)
        running_times.append(run_time)
    median_estimate = round(statistics.median(estimates))
    median_time = round(sum(running_times) / R)
    
    # Print outputs
    print("Dataset = ", data_path)
    print("Number of edges = ", numedges)
    print('Number of colors = ', C)
    print('Number of Repetitions  = ', R)
    print('Approximation through node coloring')
    print('- Number of triangles (median over ', R, ' runs) = ', median_estimate)
    print("- Average running time (median over ", R, " runs) = ", median_time, 'ms')   
    """
    

if __name__ == "__main__":
    main()