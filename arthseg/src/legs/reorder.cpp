#include "leg_utils.hpp"
#include "legs.hpp"

using LegPair = std::pair<std::vector<Point>, std::vector<Point>>;

static Point get_centroid(const std::vector<Point> &component)
{
    size_t row = 0, col = 0;
    for (const auto &point : component) {
        row += point.row;
        col += point.col;
    }
    return { row /= component.size(), col /= component.size() };
}

void reored_legs(PyArrayObject *image, PyObject *labels, PyObject *pair_labels, PyObject *body_labels, const std::vector<std::vector<Point>> &legs)
{
    std::vector<std::pair<std::vector<Point>, Point>> left, right;

    for (const auto &leg : legs) {
        auto centroid = get_centroid(leg);
        auto leg_start = get_centroid(find_leg_start(image, body_labels, leg));

        if (centroid.col < leg_start.col) {
            left.push_back({ leg, centroid });
        } else {
            right.push_back({ leg, centroid });
        }
    }
}

// static std::vector<LegPair> make_pairs(std::vector<std::pair<std::vector<Point>, Point>> &left, std::vector<std::pair<std::vector<Point>, Point>> &right)
// {
//     std::vector<LegPair> pairs;

//     bool left_full = left.size() == 4;
//     bool right_full = right.size() == 4;

//     auto l = left.begin();
//     auto r = right.begin();

//     while ((!left.empty() || !right.empty()) && pairs.size() < 4) {
//         if (left.empty()) {
//             pairs.push_back({}, r->first);
//             r++;
//         } else if (right.empty()) {
//             pairs.push_back(l->first, {});
//             l++;
//         } else {
//             if ((left_full && right_full) || ((r + 1 == right.end() || abs(l->second - r->second) < abs(l->second - (r + 1)->second)) && (l + 1 == left.end() || abs(l->second - r->second) < abs((l + 1)->second - r->second)))) {
//                 pairs.emplace_back(*l, *r);
//                 l++;
//                 r++;
//             } else if (left_full || (l->second < r->second && !right_full)) {
//                 pairs.emplace_back(*l, nullptr);
//                 l++;
//             } else {
//                 pairs.emplace_back(nullptr, *r);
//                 r++;
//             }
//         }
//     }

//     return pairs;
// }
