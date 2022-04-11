import collections
import subprocess
from sys import stderr

import cv2
import numpy as np

SAT_SOLVER_PATH = "/home/wenjie/source/minisat/simp/minisat_static"


class SatSpace:
    def __init__(self):
        self.count = 0
        self.clauses = []

    def get_next(self):
        self.count += 1
        return self.count

    def add_clause(self, clause):
        self.clauses.append(clause)

    def output_to_cnf(self, cnf):
        cnf.write(f"p cnf {self.count} {len(self.clauses)}\n")
        for clause in self.clauses:
            cnf.write(" ".join(str(lit) for lit in clause) + " 0\n")

    def read_result(self, f):
        self.result = np.zeros(self.count, dtype=bool)
        for line in f:
            line = line.strip()
            if line == "SAT":
                continue
            line = line.split(" ")
            for lit in line:
                lit = int(lit)
                if lit < 0:
                    self.result[-lit - 1] = False
                else:
                    self.result[lit - 1] = True

    def get_value(self, idx):
        return self.result[idx - 1]


class GraphPuzzle:
    def __init__(self, puzzle, space, fromr, fromc, tor, toc):
        self.fromr, self.fromc, self.tor, self.toc = fromr, fromc, tor, toc
        self.puzzle = puzzle
        self.space = space
        self.rows, self.cols = rows, cols = puzzle.shape
        self.n = n = self.rows * self.cols
        self.neigh = neigh = collections.defaultdict(set)
        for r in range(rows):
            for c in range(cols):
                if puzzle[r, c]:
                    continue
                if r + 1 < rows and not puzzle[r + 1, c]:
                    neigh[r * cols + c].add((r + 1) * cols + c)
                    neigh[(r + 1) * cols + c].add(r * cols + c)
                if c + 1 < cols and not puzzle[r, c + 1]:
                    neigh[r * cols + c].add(r * cols + c + 1)
                    neigh[r * cols + c + 1].add(r * cols + c)
        self.puzzle = puzzle = np.reshape(puzzle, (n,))
        self.s = [
            space.get_next() if not puzzle[alpha] else 0
            for t in range(n)
            for alpha in range(n)
        ]

    def get_s_loc(self, t, alpha):
        result = self.s[t * self.n + alpha]
        if result == 0:
            raise Exception
        return result

    def gen_clauses(self):
        n = self.n
        cols = self.cols
        fromr, fromc, tor, toc = self.fromr, self.fromc, self.tor, self.toc
        from_point = fromr * cols + fromc
        to_point = tor * cols + toc
        puzzle = self.puzzle
        neigh = self.neigh

        for alpha in range(n):
            if not puzzle[alpha]:
                if alpha == from_point:
                    self.space.add_clause([self.get_s_loc(0, alpha)])
                else:
                    self.space.add_clause([-self.get_s_loc(0, alpha)])

        self.space.add_clause([self.get_s_loc(n - 1, to_point)])

        for t in range(n - 1):
            for alpha in range(n):
                if not puzzle[alpha]:
                    self.space.add_clause(
                        [
                            -self.get_s_loc(t + 1, alpha),
                            self.get_s_loc(t, alpha),
                            *(self.get_s_loc(t, beta) for beta in neigh[alpha]),
                        ]
                    )

    def trace_route(self):
        cols = self.cols
        n = self.n
        neigh = self.neigh

        fromr, fromc, tor, toc = self.fromr, self.fromc, self.tor, self.toc
        from_point = fromr * cols + fromc
        to_point = tor * cols + toc

        if not self.space.get_value(self.get_s_loc(n - 1, to_point)):
            raise Exception
        route = [(tor, toc)]
        for t in range(n - 1, -1, -1):
            if self.space.get_value(self.get_s_loc(t, to_point)):
                continue
            has_to_point = False
            for alpha in neigh[to_point]:
                if self.space.get_value(self.get_s_loc(t, alpha)):
                    route.append((alpha // cols, alpha % cols))
                    to_point = alpha
                    has_to_point = True
                    break
            if not has_to_point:
                raise Exception
        return route


def main():
    sat = SatSpace()
    puzzle = """0111100110
                0001101010
                1000000000
                1001110001
                0000001100
                1111101100
                0010000010
                0001010110
                0110001110
                0100110010"""
    puzzle = """01100100001000100100
                00000010111101000001
                01000111001000000001
                00100010011101010001
                01100001001111000101
                10000001111000000100
                00010000001100100101
                01100101001000001111
                00100101100010111111
                01100010000000000001
                10011011000101101111
                01101100110000100001
                00000011111110010100
                01111101110000100101
                01010000100011000111
                11000100101101010000
                01100001110000000100
                00001011110011000010
                01111011110111010100
                10000111110110001100"""
    puzzle = np.array(
        [[int(ch) for ch in line.strip()] for line in puzzle.splitlines()], dtype=bool,
    )
    rows, cols = puzzle.shape
    graph = GraphPuzzle(puzzle, sat, 0, 0, rows - 1, cols - 1)
    graph.gen_clauses()
    with open("sat.cnf", "w") as fout:
        sat.output_to_cnf(fout)
    run = subprocess.run(
        [SAT_SOLVER_PATH, "sat.cnf", "out.cnf"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if not b"SATISFIABLE" in run.stdout:
        raise Exception("Failed to call SAT solver")
    with open("out.cnf") as fin:
        sat.read_result(fin)
    result = graph.trace_route()
    result = result[::-1]

    print(" -> ".join(map(lambda x: f"{x[0]},{x[1]}", result)))

    image_scale = 15
    thickness = 1
    middle_offset = (image_scale - 1) // 2
    image = np.zeros((rows * image_scale, cols * image_scale, 3), dtype=np.uint8)
    for c in range(cols):
        for r in range(rows):
            image[
                c * image_scale : c * image_scale + image_scale,
                r * image_scale : r * image_scale + image_scale,
                :,
            ] = (255 - 255 * puzzle[c, r])
    for (r1, c1), (r2, c2) in zip(result, result[1:]):
        cv2.line(
            image,
            (c1 * image_scale + middle_offset, r1 * image_scale + middle_offset),
            (c2 * image_scale + middle_offset, r2 * image_scale + middle_offset),
            (0, 0, 255),
            thickness=thickness,
        )
    cv2.imwrite("out.png", image)


if __name__ == "__main__":
    main()
