


def multi_param(*inputs):
    for input in inputs:
        print(input)


multi_param(1, *[2,3,4])