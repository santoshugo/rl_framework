 Changes in framework:
 - distance in graphenv should not be predefined as euclidean. Haversine also possible?
 - remove edges, nodes that are used to create graph
 - refactor environment. split graph and abstract env. graph should be a function that returns a nx graph