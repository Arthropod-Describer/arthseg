#include <list>
#include <vector>

#include "skeletonization.hpp"

static void thinning(Matrix<bool> &marker, std::list<Point> &nodes, int iter);
static bool is_removable(const Matrix<bool> &marker, size_t row, size_t col, int iter);

std::vector<Point> skeletonization(PyArrayObject *image, const std::vector<Point> &points)
{
    Matrix<bool> marker(PyArray_DIM(image, 0), PyArray_DIM(image, 1), false);
    std::list<Point> skeleton(points.begin(), points.end());

    for (auto &point : points) {
        marker.at(point) = true;
    }

    size_t previous;
    do {
        previous = skeleton.size();
        thinning(marker, skeleton, 0);
        thinning(marker, skeleton, 1);
    } while (previous != skeleton.size());

    return { std::make_move_iterator(skeleton.begin()), std::make_move_iterator(skeleton.end()) };
}

static void thinning(Matrix<bool> &marker, std::list<Point> &nodes, int iter)
{
    std::vector<Point> removed_stack;

    for (auto node = nodes.begin(); node != nodes.end(); node++) {
        if (is_removable(marker, node->row, node->col, iter)) {
            removed_stack.emplace_back(std::move(*node));
            node = --nodes.erase(node);
        }
    }

    for (auto &point : removed_stack) {
        marker.at(point) = false;
    }
}

static bool is_removable(const Matrix<bool> &marker, size_t row, size_t col, int iter)
{
    bool rowm = row > 0;
    bool rowp = row + 1 < marker.rows;
    bool colm = col > 0;
    bool colp = col + 1 < marker.cols;

    bool p1 = rowm && marker.at(row - 1, col);
    bool p2 = rowm && colp && marker.at(row - 1, col + 1);
    bool p3 = colp && marker.at(row, col + 1);
    bool p4 = rowp && colp && marker.at(row + 1, col + 1);
    bool p5 = rowp && marker.at(row + 1, col);
    bool p6 = rowp && colm && marker.at(row + 1, col - 1);
    bool p7 = colm && marker.at(row, col - 1);
    bool p8 = rowm && colm && marker.at(row - 1, col - 1);

    int A = (!p1 && p2) + (!p2 && p3) + (!p3 && p4) + (!p4 && p5) +
            (!p5 && p6) + (!p6 && p7) + (!p7 && p8) + (!p8 && p1);
    int B = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
    bool m1 = iter == 0 ? (p1 * p3 * p5) : (p1 * p3 * p7);
    bool m2 = iter == 0 ? (p3 * p5 * p7) : (p1 * p5 * p7);

    return A == 1 && (B >= 2 && B <= 6) && !m1 && !m2;
}
