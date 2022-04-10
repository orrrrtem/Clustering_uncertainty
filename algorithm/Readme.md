## Algorithm API

Input:

*  **graph** - *adjacency matrix represented via numpy array with shape* ($|V|$, $|V|$). Input graph shall follow **completeness** property that is no zero values in the adjacency matrix.
*  **num_clusters** - *integer*

Output: 

*   **labels** - *numpy int array*

Example:  
Two connected vertices are partitioned into two clusters.
```
print(algo(np.array([1, 0.5], [0.5, 1]), num_clusters = 2))  
-> [0, 1]
```