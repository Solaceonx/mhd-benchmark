"""Write the unit square as NE x NE quadrilaterals in MFEM's mesh format (v1.0): python3 make_mesh.py NE FILE

Boundary attributes as MHD-ALE's test cases expect them: 1 y = 0, 2 x = 1, 3 y = 1, 4 x = 0. MHD-ALE makes the mesh
periodic itself (the directions the test case names), so the file always has all four sides.
"""
import sys


def main(ne, path):
    ne = int(ne);v = lambda i, j: i + (ne + 1)*j
    out = ["MFEM mesh v1.0", "", "dimension", "2", "", "elements", str(ne*ne)]
    out += [f"1 3 {v(i, j)} {v(i + 1, j)} {v(i + 1, j + 1)} {v(i, j + 1)}" for j in range(ne) for i in range(ne)]
    out += ["", "boundary", str(4*ne)]
    out += [f"1 1 {v(i, 0)} {v(i + 1, 0)}" for i in range(ne)]
    out += [f"2 1 {v(ne, j)} {v(ne, j + 1)}" for j in range(ne)]
    out += [f"3 1 {v(i + 1, ne)} {v(i, ne)}" for i in range(ne)]
    out += [f"4 1 {v(0, j + 1)} {v(0, j)}" for j in range(ne)]
    out += ["", "vertices", str((ne + 1)**2), "2"]
    out += [f"{i/ne!r} {j/ne!r}" for j in range(ne + 1) for i in range(ne + 1)]
    with open(path, "w") as f:f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
