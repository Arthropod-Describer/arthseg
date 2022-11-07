#include "legs.hpp"
#include "shortest_path.hpp"
#include "utils.cpp"

std::vector<std::vector<Point>> split_leg(PyArrayObject *image, PyObject *body_labels, const Component &component)
{
    std::vector<Point> start = find_leg_start(image, body_labels, component.nodes);

    auto sorted = shortest_path(image, component.nodes, start);

    size_t label = 1;
    Matrix<size_t> group_labels(PyArray_DIM(image, 0), PyArray_DIM(image, 1));
    Matrix<bool> marker(PyArray_DIM(image, 0), PyArray_DIM(image, 1));
    std::vector<std::vector<Point>> groups;

    for (auto &point : sorted) {
        marker.at(point) = true;
    }

    for (auto it = sorted.rbegin(); it != sorted.rend(); it++) {
        auto &node = *it;
        if (group_labels.at(node) == 0) {
            group_labels.at(node) = label;
            groups.push_back({ node });
            label++;
        }

        for (size_t i = 0; i < 8; i++) {
            auto row = node.row + drow[i];
            auto col = node.col + dcol[i];
            if (is_outside(image, row, col) || !marker.at(row, col) || group_labels.at(row, col) == group_labels.at(node)) {
                continue;
            }

            auto index = group_labels.at(row, col) - 1;

            if (group_labels.at(row, col) == 0) {
                group_labels.at(row, col) = group_labels.at(node);
                groups[group_labels.at(node) - 1].push_back({ row, col });
            } else if (groups[index].size() < sorted.size() / 10 || groups[group_labels.at(node) - 1].size() < sorted.size() / 10) {
                for (auto &point : groups[index]) {
                    group_labels.at(point) = group_labels.at(node);
                    groups[group_labels.at(node) - 1].push_back(point);
                }
                groups[index].clear();
            }
        }
    }

    return groups;
}