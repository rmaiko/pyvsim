import numpy as np
from Object import Object
from pprint import pprint

points = [[0,0,0],
		  [1,0,0],
		  [1,1,0],
		  [0,1,0],
		  [0,0,1],
		  [1,0,1],
		  [1,1,1],
		  [0,1,1]]

# mixed normals
# conn = [[0,1,3],[1,2,3], #xy-plane +z inwards
		# [0,1,4],[1,5,4], #xz-plane -y outwards
		# [4,5,7],[5,6,7], #xy-plane +z outwards
		# [3,6,7],[3,2,6], #xz-plane -y inwards
		# [1,5,6],[1,6,2], #yz-plane -x inwards
		# [0,3,7],[0,7,4]] #yz-plane +x inwards

# normals pointing outside
# conn = [[5,7,4],[5,6,7], # normal +z
	   # [3,2,1],[0,3,1], # normal -z
	   # [3,6,2],[6,3,7], # normal +y
	   # [1,5,4],[4,0,1], # normal -y
	   # [5,1,6],[1,2,6], # normal +x
	   # [7,0,4],[7,3,0]] # normal -x

# normals pointing inside
conn = [[4,7,5],[7,6,5], # normal -z
		[1,2,3],[1,3,0], # normal +z
		[2,6,3],[7,3,6], # normal -y
		[4,5,1],[1,0,4], # normal +y
		[6,1,5],[6,2,1], # normal -x
		[4,0,7],[0,3,7]] # normal +x
				   
mesh = Object()
mesh.points = np.array(points)
mesh.connectivity = np.array(conn)

pprint(mesh.points[mesh.connectivity])
pprint(mesh.points[mesh.connectivity][:,1])
pprint(mesh.points[mesh.connectivity][:,1]-np.array([1,1,1]))    
pprint(np.cross(np.array([[1,0,0],[0,1,0]]),np.array([[0,1,0],[0,0,1]])))
pprint(np.sum(np.array([[1,0,0],[0,1,0]])*np.array([[0,1,0],[0,0,1]]),1))