class Node:
    def __init__(self, state, action, parent):
        self.state = state
        self.action = action
        self.parent = parent

#DFS   
class StackFrontier():
    def __init__(self):
        self.frontier = []
        
    def add(self, node):
        self.frontier.append(node)
        
    def remove(self):
        if len(self.frontier) == 0:
           raise Exception("frontier is empty!")
        node = self.frontier[-1]
        self.frontier = self.frontier[:-1]
        return node
    
    def contains_state(self, state):
        for node in self.frontier:
            if node.state == state:
                return True
        return False
        # return any(node.state == state for node in self.frontier)
    
    def empty(self):
        return len(self.frontier) == 0
   
# BFS 
class QueueFrontier(StackFrontier):
    def remove(self):
        if len(self.frontier) == 0:
            raise Exception("frontier is empty!")
        node = self.frontier[0]
        self.frontier = self.frontier[1:]
        return node
    
class Maze:
    def __init__(self, filename):
        # Read file
        self.filename = filename
        with open(filename) as f:
            contents = f.read()
    
    
    