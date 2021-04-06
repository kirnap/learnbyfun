# breadth first search path finding
# Given graph, what is the path from start to end?
#

# given following graph
graph = {
    '1': ['2', '3', '4'],
    '2': ['5', '6'],
    '5': ['9', '10'],
    '4': ['7', '8'],
    '7': ['11', '12']
}


def bfs(g, s, e):
    """
    :param g:graph
    :param s: start idx
    :param e: end idx
    :return: path from start to end
    """
    q = [[s]]  # initialize queue

    while q:
        path = q.pop(0)

        # get the last state from path
        state = path[-1]
        if state == e:
            return path
        for adjacent in g.get(state, []):
            new_path = list(path)
            new_path.append(adjacent)
            q.append(new_path)
        omer = 3

if __name__ == '__main__':
    print(bfs(graph, '1', '11'))