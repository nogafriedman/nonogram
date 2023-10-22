#############################################
# FILE : nonogram
# WRITER : Noga_Friedman , nogafri , 209010479
# Exercise: ex8
# DESCRIPTION: a program that solves nonograms
# STUDENTS I DISCUSSED THE EXERCISE WITH: -
# WEB PAGES I USED: -
#############################################

### task answer: ###
# I chose to leave undecided squares as neutral (-1) - filling them as 1/0
# is untrue to the constraints given and might lead to future false deductions.
###################

def constraint_satisfactions(n, blocks):  # 1
    """
    calculates all possible solutions of coloring a row with length n by
    given blocks (block = number of consecutive squares to color)
    :param n: int - row length
    :param blocks: list where each element is a block as an int
    :return: list of lists of length n, with elements 0/1 in each sublist
    """
    row = initiate_row(n)
    possible_solutions = []
    color_block(n, row, blocks, possible_solutions)
    is_end_of_row(n, possible_solutions)
    return possible_solutions


def initiate_row(n):
    """
    initiates a row with '0' n times (uncolored squares) in order to run
    through it and color the squares later according to the blocks given
    :param n: int - length of the row
    :return: string with length n where each char is a '0'
    """
    row = '0' * n
    return row


def num_colored_squares(blocks):
    """
    checks how many "colored squares" (marked '1') should be in the row in the
    end of the coloring according to the parameter 'blocks'.
    :param blocks: list where each element is a block as an int
    :return: int - number of squares that should be colored ('1') in the
    end of the solution calculation
    """
    colored = 0
    for num in blocks:
        colored += num
    return colored


def initiate_block_with_space(block):
    """
    creates a string of the block when colored and adds '0' for a space
    between blocks
    :param block: list where each element is a block as an int
    :return: colored block + empty square for space
    """
    full_block = block * '1' + '0'
    return full_block


def color_block(n, row, blocks, lst, row_ind=0, block_ind=0):  # recursive func
    """
    calls itself recursively and adds to the solution list all the possible
    colorings for the given length and blocks
    :param n: int - row length
    :param row: string - row of '0' as initiated in initiate_row
    :param blocks: list where each element is a block as an int
    :param lst: solution list, starts as an empty list
    :param row_ind: current row index - updates with each recursion
    :param block_ind: current index of the block handled
    :return: None
    """
    if block_ind == len(blocks):  # if finished going through all blocks
        lst.append([int(num) for num in row])
        return
    if not check_future_blocks(row, blocks, row_ind, block_ind):
        return  # backtrack
    empty_row = row  # creates a 'copy' of the row for the second recursion
    full_block = initiate_block_with_space(blocks[block_ind])
    # replaces the '?' with the colored block and moves an indexes forward
    row = row[:row_ind] + full_block + row[row_ind + len(full_block):]
    color_block(n, row, blocks, lst, row_ind + len(full_block), block_ind + 1)
    color_block(n, empty_row, blocks, lst, row_ind + 1, block_ind)


def check_future_blocks(row, blocks, row_ind, block_ind):
    """
    checks the validity of the current coloring by calculating whether
    there are enough squares in the rest of the row for the uncolored blocks
    :param row: current version of the row with previous colorings
    :param blocks: list of blocks
    :param row_ind: current index in row
    :param block_ind: current block handled
    :return: True if valid, False if unsolvable
    """
    blocks_sum = 0
    blocks_count = 0
    for i in range(block_ind, len(blocks)):
        blocks_sum += blocks[i]
        blocks_count += 1
    required_squares = blocks_sum + blocks_count - 1
    len_row_left = len(row) - row_ind
    if len_row_left < required_squares:
        return False
    return True


def is_end_of_row(n, lst):
    """
    checks if current block handled is in the end of the row, meaning the
    '0' in the end of it as initiated in initiate_block is redundant,
    then removes it
    :param n: int - row length
    :param lst: list of lists- all found solutions
    :return: updated list
    """
    for sublist in lst:
        if len(sublist) > n:
            sublist.pop()
    return lst


####################

def row_variations(row, blocks):  # 2
    """
    calculates all possible solutions for coloring a row by given blocks
    (block = number of consecutive squares to color),
    with specific squares in given row set to be colored or empty
    :param row: list where each element is either 1, -1 or 0 (1 = colored
    square, -1 = undecided, 0 = empty square)
    :param blocks: list where each element is a block as an int
    :return: list of lists of length n, with elements 0/1 in each sublist
    """
    lst = []
    wanted_sum = num_colored_squares(blocks)
    num_colored, num_neutral = count_squares(row)
    recursive_variations(row.copy(), blocks, lst, wanted_sum, num_colored,
                                                       num_neutral)
    return lst  # possible solutions


def count_squares(row):
    """
    counts the amount of colored/neutral squares in the starting row.
    :param row: current row given
    :return: int, int - number of colored squares (1), number of neutral
    squares (-1)
    """
    num_colored = 0
    num_neutral = 0
    for square in row:
        if square == 1:
            num_colored += 1
        if square == -1:
            num_neutral += 1
    return num_colored, num_neutral


def recursive_variations(row, blocks, lst, wanted_sum, num_colored,
                         num_neutral, row_ind=0, block_ind=0):
    """
    recursively colors the given row according to the blocks
    :param row: list where each element is either 1, -1 or 0
    :param blocks: list where each element is a block as an int
    :param lst: list that starts empty and gets the solutions as sublists
    :param wanted_sum: the sum of colored blocks in the final solution
    :param num_colored: number of currently colored squares
    :param num_neutral: number of currently neutral squares
    :param row_ind: current index in row
    :param block_ind: current block to color
    :return: lst - a list of solutions as sublists
    """
    if block_ind == len(blocks):  # if finished going through all blocks
        for i in range(row_ind, len(row)):
            if row[i] == 1:
                return
        lst.append([num if num != -1 else 0 for num in row])
        return
    # checks if the row with the block inserted exceeds the boundaries:
    if row_ind + blocks[block_ind] > len(row):
        return  # backtrack
    if num_colored > wanted_sum or num_neutral + num_colored < wanted_sum:
        return  # backtrack
    if row[row_ind] == 1 or row[row_ind] == -1:  # color the block
        flag, colored, neutrals = check_block_fit(row, blocks, row_ind,
                                                block_ind)
        if flag:
            full_block = initiate_block(blocks, block_ind)
            if row_ind + blocks[block_ind] == len(row):  # if reached the
                # end of the row
                new_row = row[:row_ind] + full_block + row[row_ind + blocks[
                    block_ind] + 1:]
            else:
                new_row = row[:row_ind] + full_block + [0] + row[row_ind +
                     blocks[block_ind] + 1:]  # new row with the block colored
            recursive_variations(new_row, blocks, lst, wanted_sum,
                                 num_colored + blocks[block_ind] - colored,
                                num_neutral - neutrals, row_ind + blocks[
                                     block_ind] + 1, block_ind + 1)
    if row[row_ind] == 0:
        recursive_variations(row, blocks, lst, wanted_sum, num_colored,
                             num_neutral, row_ind + 1, block_ind)
    if row[row_ind] == -1:
        row[row_ind] = 0
        recursive_variations(row, blocks, lst, wanted_sum, num_colored,
                             num_neutral - 1, row_ind + 1, block_ind)


def initiate_block(blocks, block_ind):
    """
    creates a list of how to finished block should look like after colored
    :param blocks: list where each element is a block as an int
    :param block_ind: current block index
    :return: a list where each element is an int 1 for a colored square
    """
    full_block = blocks[block_ind] * [1]
    return full_block


def check_block_fit(row, blocks, row_ind, block_ind):
    """
    checks if there are enough remaining squares for the current block
    :param row: current state of the row
    :param blocks: list with ints
    :param row_ind: current row index
    :param block_ind: current block index
    :return: True if the block fits in the remaining squares, False if it
    doesn't
    """
    colored = 0
    neutrals = 0
    for i in range(0, blocks[block_ind]):
        if row[row_ind + i] == 0:  # can't insert block
            return False, None, None
        if row[row_ind + i] == 1:  # amount of squares that are already colored
            colored += 1
        if row[row_ind + i] == -1:  # amount of new squares to color
            neutrals += 1
    # checks if the row with the block inserted exceeds the boundaries:
    if row_ind + blocks[block_ind] == len(row):
        return True, colored, neutrals
    # checks if the row with the block inserted has too many squares colored:
    if row[row_ind + blocks[block_ind]] == 1:
        return False, None, None
    return True, colored, neutrals


def check_future_variations(row, wanted_sum):
    """
    runs various checks in order to determine whether the current solution
    is valid- checks if there are enough future squares in order to finish
    the coloring with the right amount of colored blocks
    :param row: current state of the row
    :param wanted_sum: int - sum of final filled blocks according to constraints
    :return: True if passed the checks, False if didn't (then backtracks)
    """
    count_colored = 0  # number of squares with '1'
    count_undecided = 0  # number of undecided squares with '-1'
    for i in range(len(row)):
        if row[i] == 1:
            count_colored += 1
        if row[i] == -1:
            count_undecided += 1
    if count_colored > wanted_sum:  # if number of colored squares is bigger
        # than the number of squares that should be colored in the solution
        return False
    if count_undecided + count_colored < wanted_sum:  # if total amount of
        # squares that can be colored is not enough for the solution
        return False
    return True


####################

def intersection_row(rows):  # 3
    """
    gets various rows and returns the constraints they have in common
    :param rows: list of lists, each sub-list is a row, all with the same length
    :return: a list with the common constraints (1, -1 or 0)
    """
    intersection_list = []
    if len(rows) == 0:
        return intersection_list
    for i in range(len(rows[0])):
        same_num = True
        for r in range(len(rows)):
            if rows[r][i] != rows[0][i]:
                same_num = False
        if same_num:
            intersection_list.append(rows[0][i])
        else:
            intersection_list.append(-1)
    if intersection_list == []:
        return []
    else:
        return intersection_list


####################
# PART 3 #

def solve_easy_nonogram(constraints):  # 4
    """
    gets a list of constraints and solves a nonogram until there's no new
    deductions that can be made.
    :param constraints: a list containing two lists: one for the row
    constraints and one for the column constraints.
    each list contains sublists, that represent the number of blocks that
    should be colored in the respective row/column.
    the number of sublists equals the number of rows and column on the board.
    example: [ [ [2, 1], [4], [1, 3] ] #rows , [ [2], [4], [3, 1] #columns ] ]
    :return: completed board (after all colorings done) / None
    """
    is_empty = check_empty_constraints(constraints)
    if is_empty[0]:
        return is_empty[1]

    board = initiate_board(constraints)  # creates a board of the appropriate
    # size, with neutral squares (-1)

    updated_board = run_through_rows(constraints, board, 0)  # runs through
    # all rows to find intersections for the constraints
    if not updated_board:  # unsolvable
        return
    transposed_board = transpose_board(updated_board)  # turn cols into rows
    changes = set()

    for r in range(len(transposed_board)):  # run through all transposed cols
        transposed_board[r] = run_through_single_row(constraints[1][r],
                                                  transposed_board[r], changes)
        if not transposed_board[r]:  # if received []
            return
    updated_board = transpose_board(transposed_board)  # transpose back to rows
    while len(changes) != 0:  # run until no more new changes can be made
        new_changes = set()  # set to add the indexes where changes occurred
        for change in changes:  # fill in according to constraints
            updated_board[change] = run_through_single_row(constraints[0][change],
                                     updated_board[change], new_changes)
            if not updated_board[change]:  # if received []
                return
        changes = new_changes
        new_changes = set()
        for change in changes:  # run through columns
            transposed_row = transpose_col(updated_board, change)
            transposed_row = run_through_single_row(constraints[1][change],
                                                transposed_row, new_changes)
            if not transposed_row:  # if received []
                return
            add_row_as_col_to_board(updated_board, transposed_row, change)
        changes = new_changes
    return updated_board


def initiate_board(constraints):
    """
    creates a board with the appropriate size according to the constraints
    given.
    :param constraints: list of two lists, each containing sublists as the
    number of the rows/column in the current wanted board
    :return: list of lists, each list representing a row in the board
    """
    board = []
    for i in range(len(constraints[0])):
        board.append([-1] * len(constraints[1]))
    return board


def run_through_rows(constraints, board, constraints_ind):
    """
    runs through all the rows of the given board and returns an updated
    board with the intersections of all possible solutions colored.
    :param constraints:
    :param board:
    :param constraints_ind: int - 0 if the function is called to go through
    the rows, 1 if the function is called to go through the columns (as
    transposed)
    :return: updated board - list of lists, each list is a row
    """
    updated_board = []
    for i in range(len(board)):
        # go through each row and find possible solutions
        possible_solutions = row_variations(board[i], constraints[
            constraints_ind][i])
        if len(possible_solutions) == 0:
            return
        # check for squares with the same solution in all options
        intersected_solution = intersection_row(possible_solutions)
        if len(intersected_solution) == 0:
            return
        updated_board.append(intersected_solution)
    return updated_board


def run_through_single_row(row_constraint, transposed_row, changes):
    """
    runs only through the rows where a change was made, meaning other
    options to color could be found
    :param row_constraint: the constraint/block of the row
    :param transposed_row: original row to run through
    :param changes: set of indexes to look for options to color
    :return: updated board, changes (set)
    """
    possible_solutions = row_variations(transposed_row, row_constraint)
    intersected_row = intersection_row(possible_solutions)
    if not intersected_row:
        return
    find_changes(transposed_row, intersected_row, changes)
    return intersected_row


def find_changes(og_row, new_row, col_changes):
    """
    runs through all the rows on the board and checks if they changed after
    running through the intersection function.
    if a change was made, the column index of the change is added into a set.
    :param og_row: list of the original row before going through changes
    :param new_row: updated row received from the intersection function
    :param col_changes: set
    :return: None
    """
    for col_ind in range(len(og_row)):
        if og_row[col_ind] != new_row[col_ind]:
            col_changes.add(col_ind)


def transpose_board(board):
    """
    receives the updated board with the changes made to the rows,
    takes the columns and puts them as rows in order to run through them
    and find their intersections.
    :param board: list of lists, each list is a row on the board
    :return: a transposed version of board
    """
    transposed_board = [[] for _ in range(len(board[0]))]
    for row in board:
        for ind in range(len(row)):
            transposed_board[ind].append(row[ind])
    return transposed_board


def transpose_col(board, col_ind):
    """
    transposes only a specific column into a row
    :param board: list of lists - current version of the board
    :param col_ind: int - wanted column index
    :return: transposed version of the column - list of ints
    """
    transposed_row = []
    for row in board:
        transposed_row.append(row[col_ind])
    return transposed_row


def add_row_as_col_to_board(board, transposed_row, change):
    """
    adds the transposed row back to the original column on the board.
    :param board: current version of the board (before changes)
    :param transposed_row: updated version of the column after intersections
    :param change: int - column index on the board
    :return: updated board
    """
    for row_ind in range(len(board)):
        board[row_ind][change] = transposed_row[row_ind]
    return board


########################

def solve_nonogram(constraints):  # 5
    """
    solves a nonogram with deductions (by calling solve_easy_nonogram) or
    with guesses, and returns all of the possible final solutions.
    :param constraints: list of two sublists- one for the rows and one for
    the columns, each contains more sublists with the blocks.
    :return: list of possible solutions (final boards)
    """
    final_boards = []
    board = solve_easy_nonogram(constraints)
    if board is None:  # unsolvable board
        return final_boards
    if len(board) == 0:  # board with 0 rows
        return [[]]

    neutral_row_indexes = find_neutral_squares(board)  # goes through the
    # board and looks for rows with neural squares
    # go into recursive function:
    try_row_solution(board, constraints, neutral_row_indexes, final_boards)
    return final_boards


def try_row_solution(board, constraints, ind_list, final_boards,
                     neutral_ind=0):
    """
    recursive function for solve_nonogram. for every row with neutral
    squares (-1), tries all the possible solutions of it until gets a
    contradiction or solves the board.
    :param board: list of lists
    :param constraints: list of two lists, each list contains sublists
    representing blocks to color
    :param ind_list: list of ints, each int is an index of a row with -1
    :param final_boards: list that starts empty and gets appended the final
    solution found in the recursion
    :param neutral_ind: int from ind_list
    :return: None
    """
    if neutral_ind == len(ind_list):
        final_boards.append(board)  # add final solution found into list
        return  # stopping condition
    row_ind = ind_list[neutral_ind]
    possible_solutions = row_variations(board[row_ind], constraints[0][
        row_ind])
    if not possible_solutions:  # unsolvable
        return  # unsolvable - backtrack
    for row_solution in possible_solutions:
        board_copy = board.copy()
        board_copy[row_ind] = row_solution
        col_changes_indexes = check_col_changes(board, board_copy, row_ind)
        if not check_solution(board_copy, col_changes_indexes, constraints):
            return  # unsolvable - backtrack
        try_row_solution(board_copy, constraints, ind_list, final_boards,
                         neutral_ind + 1)


def check_col_changes(og_board, new_board, row_ind):
    """
    compare the original board with the new updated board and check which
    columns had changes in them.
    :param og_board: list of lists
    :param new_board: list of lists
    :param row_ind: int - index of the row where changes were made
    :return: list of ints
    """
    col_changes_indexes = []
    for col_ind in range(len(og_board[row_ind])):
        if og_board[row_ind][col_ind] != new_board[row_ind][col_ind]:
            col_changes_indexes.append(col_ind)
    return col_changes_indexes


def check_solution(board, col_changes_indexes, constraints):
    """
    check if the row solution works with the column constraints and doesn't
    make a contradiction.
    :param board: list of lists - current board
    :param col_changes_indexes: list of ints of the columns with the changes
    :param constraints: list of lists
    :return: True if the solution works, False if it makes a contradiction
    """
    valid = True
    for col_ind in col_changes_indexes:
        col_as_row = transpose_col(board, col_ind)
        possible_solutions = row_variations(col_as_row, constraints[1][
            col_ind])
        if not possible_solutions:
            valid = False  # the row solution makes a contradiction
        # if found only one possible solution, update the column as the
        # solution found:
        if len(possible_solutions) == 1:
            copy_board(board)
            add_row_as_col_to_board(board, possible_solutions[0], col_ind)
    return valid


def copy_board(board):
    """
    make a "copy" of the board with different pointers to the rows.
    :param board: list of lists
    :return: None
    """
    for row_ind in range(len(board)):
        board[row_ind] = board[row_ind].copy()


def find_neutral_squares(board):
    """
    runs through the rows of the board and check for neutral squares that
    are yet to be colored.
    :param board: list of lists
    :return: list with ints as elements, each int is an index of a row
    """
    neutral_row_indexes = []
    for row_ind in range(len(board)):
        if -1 in board[row_ind]:
            neutral_row_indexes.append(row_ind)
    return neutral_row_indexes


def check_empty_constraints(constraints):
    """
    returns the appropriate board for empty constraint cases.
    :param constraints: list of two lists- for rows and for columns,
    each contains lists that represent the blocks
    :return: list representing the board
    """
    if len(constraints[0]) == 0:
        return [True, []]  # return empty board
    if len(constraints[1]) == 0:
        empty = True
        for row in constraints[0]:
            if row != []:  # contradiction - return None
                empty = False
        if empty:  # if True
            return [True, constraints[0]]  # return the board as the row
            # constraints
    return [False]
