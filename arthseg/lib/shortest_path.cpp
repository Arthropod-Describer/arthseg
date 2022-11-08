#include <limits>
#include <queue>

#include "shortest_path.hpp"
#include "utils.hpp"

static constexpr size_t line = 10, diagonal = 14;

std::vector<Node> shortest_path(PyArrayObject *image, const std::vector<Point> &points, const std::vector<Point> &start)
{
    _import_array();
    const size_t rows = PyArray_DIM(image, 0);
    const size_t cols = PyArray_DIM(image, 1);
    Matrix<bool> marker(rows, cols);
    Matrix<size_t> distance(rows, cols, std::numeric_limits<size_t>::max());

    for (auto &point : points) {
        marker.at(point) = true;
    }

    std::vector<Node> nodes;
    std::priority_queue<Node> queue;
    for (auto &point : start) {
        queue.emplace(point.row, point.col, 0);
        distance.at(point) = 0;
    }

    while (!queue.empty()) {
        auto node = queue.top();
        queue.pop();

        if (!marker.at(node)) {
            continue;
        }

        marker.at(node) = false;
        nodes.push_back(node);
        for (size_t i = 0; i < CONNECTIVITY_8; i++) {
            auto row = node.row + drow[i];
            auto col = node.col + dcol[i];
            auto cost = node.cost + (i < 4 ? line : diagonal);

            if (row < rows && col < cols && marker.at(row, col) && cost < distance.at(row, col)) {
                distance.at(row, col) = cost;
                queue.emplace(row, col, cost);
            }
        }
    }

    return nodes;
}
