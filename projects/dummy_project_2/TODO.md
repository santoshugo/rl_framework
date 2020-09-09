 - ensure no more than 5 agvs at charging station
 - finish implementing step
 - ensure that no two agvs an be at the same position
 - ensure only one can be pickingup /dropping cart
 
 - refactor code to use agent_no as key
 - refactor code to use node ids when applicable
 
 
 Changes in framework:
 - distance in graphenvironment should not be predefined as euclidean. Haversine also possible?
 - remove edges, nodes that are used to create graph
 