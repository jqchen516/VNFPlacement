from itertools import combinations

vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
                     [2, 6, 100], [3, 4, 100], [2, 4, 100],
                     [1, 1, 100], [2, 1, 100], [2, 2, 100],
                     [2, 2, 100], [1, 6, 100]]
number_of_node = 5


def placement():
    a = "abcdefghij"
    print(list(combinations(a, 10)))


placement()
