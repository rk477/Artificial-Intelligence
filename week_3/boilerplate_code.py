
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """
    # ====> insert your code below here
    # Check if open_list is empty
    if not self.open_list:
        return None
        
    # pseudocode ====> my_index ← GetLastIndex(open_list)
    my_index = len(self.open_list) - 1

    # pseudocode ====> the_candidate ← open_list(my_index)
    next_soln = self.open_list[my_index]

    # pseudocode ====> RemoveFromOpenList(my_index)
    self.open_list.pop(my_index)

    # <==== insert your pseudo-code and code above here
    return next_soln

    # <==== insert your code above here


class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """
    # ====> insert your code below here

    # Check if open_list is empty
    if not self.open_list:
        return None
        
    # pseudocode ====> my_index ← GetFirstIndex (open_list)
    my_index = 0

    # pseudocode ====> the_candidate ← open_list(my_index)
    next_soln = self.open_list[my_index]

    # pseudocode ====> RemoveFromOpenList(my_index)
    self.open_list.pop(my_index)
        
    # <==== insert your pseudo-code and code above here
    return next_soln
    # <==== insert your code above here

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First   search.
    You need to complete this
    """
    # ====> insert your code below here
# ====> insert your pseudo-code and code below here
    if not self.open_list:
        return None
        
    # pseudocode ====> ELSE bestChild = GetMemberWithHighestQuality (openList)
     best_index = 0
    best_quality = self.open_list[0].quality

    for i in range(1, len(self.open_list)):
        current_quality = self.open_list[i].quality

        if self.minimise:
            if current_quality < best_quality:
                best_quality = current_quality
                best_index = i
        else:
            if current_quality > best_quality:
                    best_quality = current_quality
                    best_index = i

        # Remove and return the best child
    best_child = self.open_list.pop(best_index)
    return best_child

            
        # <==== insert your pseudo-code and code above here
    return next_soln
    # <==== insert your code above here

class AStarSearch(SingleMemberSearch):
    """Implementation of A Star  search.
    You need to complete this
    """
    # ====> insert your code below here
    if not self.open_list:
            return None
        
        # pseudocode ====> bestChild GetMemberWithHighestCombinedScore(openList)
    best_index = 0
    best_combined = self.open_list[0].quality + len(self.open_list[0].variable_values)

    for i in range(1, len(self.open_list)):
            current_quality = self.open_list[i].quality
            current_cost = len(self.open_list[i].variable_values)
            current_combined = current_quality + current_cost

            # Minimize combined score
            if current_combined < best_combined:
                best_combined = current_combined
                best_index = i

        # pseudocode ====>RETURN bestChild
    best_child = self.open_list.pop(best_index)
    return best_child


        # <==== insert your pseudo-code and code above here
    return next_soln
    # <==== insert your code above here
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    pass
    # ====> insert your code below here

    # <==== insert your code above here
def create_maze_depth_better():
    pass
    # ====> insert your code below here 

    # <==== insert your code above here


    
