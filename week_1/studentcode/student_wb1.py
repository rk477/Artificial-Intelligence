from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()
    
    # ====> insert your code below here

    for puzzle1 in puzzle.value_set:
        for puzzle2 in puzzle.value_set:
            for puzzle3 in puzzle.value_set:
                for puzzle4 in puzzle.value_set:
                    #set current combination
                    my_attempt.variable_values = [puzzle1, puzzle2, puzzle3, puzzle4]

                    try:
                        #test for combination
                        res = puzzle.evaluate(my_attempt.variable_values)
                        if res == 1: #correct combination
                            return my_attempt.variable_values
                    except ValueError:
                        continue

    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
     
    #loop
    for i in range(namearray.shape[0]):
        family_name = namearray[i, -6:] #gets last 6 chars
        name = "".join(family_name) #join the chars
        family_names.append(name) #append last name to family names list

    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here

    assert attempt.shape == (9, 9), "Array must be 9x9"
    # use assertions to check that the array has 2 dimensions each of size 9

    # all rows
    for i in range(9):
        slices.append(attempt[i, :])

    # all columns
    for i in range(9):
        slices.append(attempt[:, i])

    # 9 sub squares
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            slices.append(attempt[i:i+3, j:j+3].flatten())

    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list

    for slice in slices:  # easiest way to iterate over list
        
        # print(slice) - useful for debugging?

        # get number of unique values in slice

        # increment value of tests_passed as appropriate

        if len(np.unique(slice)) == 9:
            tests_passed += 1
    
    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
