from math import sqrt, atan2, pi
import math

def read_positions():
    positions = []

    with open('0.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    with open('1.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    with open('2.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    return positions

def make_column(down_index, positions, result):

    # print down_index, positions[down_index]
    down_x, down_y = None, None
    for i in range(len(positions)):
        x, y, index, active = positions[i]
        if index == down_index and active:
            down_x, down_y = x, y
            positions[i] = (x, y, index, False)

    if down_x is None or down_y is None:
        raise Exception(str(down_index) + ' not x,y')

    # print down_index, down_x, down_y

    if len(result) == 8:
        return

    active_positions = filter(lambda (x, y, index, active): active, positions)

    polar = map(lambda (x, y, index, active): (sqrt((x - down_x) ** 2 + (y - down_y) ** 2),
                                               atan2(y - down_y, x - down_x), index, active), active_positions)

    closest = sorted(polar, key=lambda (r, a, i, _): r + r * math.cos(a))
    # print down_index, closest
    closest_r, closest_a, closest_index, _ = closest[0]

    result.append(closest_index)
    make_column(closest_index, positions, result)


def make_it_square():
    positions = read_positions()
    positions = positions[2]

    xx = map(lambda (x, y): x, positions)
    yy = map(lambda (x, y): y, positions)
    positions = zip(xx, yy, list(range(len(xx))), [True for _ in range(len(xx))])

    start_columns = [65, 72, 81, 79, 80, 88, 97, 96, 95, 100, 101, 99]
    matrix = []

    for start in start_columns:
        result = [start]
        make_column(start, positions, result)
        matrix.append(result)

    result = [52, 53, 51, 50, 43, 42]
    matrix.append(result)

    with open("smap_sq.txt", "w") as fw:
        for i in range(13):
            print matrix[i]
            for j in range(len(matrix[i])):
                fw.write(str(matrix[i][j]) + ' ' + str(i) + ' ' + str(j) + '\n')

if __name__ == '__main__':
    make_it_square()