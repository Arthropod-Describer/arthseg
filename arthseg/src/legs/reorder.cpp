#include "legs.hpp"
#include "moments.hpp"
#include "utils.cpp"

using LegWithStart = std::vector<std::pair<std::vector<Point>, Point>>;
using LegPair = std::pair<std::vector<Point>, std::vector<Point>>;

static std::vector<LegPair> make_pairs(LegWithStart &left, LegWithStart &right, Py_ssize_t size);
static bool is_closer(LegWithStart::iterator left, LegWithStart::iterator right);

void reored_legs(PyArrayObject *image, PyObject *body_labels, PyObject *pair_labels, const std::vector<std::vector<Point>> &legs, const std::vector<Point> &body)
{
    LegWithStart left, right;

    auto body_moments = Moments(body);
    for (const auto &leg : legs) {
        auto centroid = Moments::get_centroid(leg);
        auto leg_start = Moments::get_centroid(find_leg_start(image, body_labels, leg));

        if (body_moments.half_axis(centroid) < 0) {
            left.push_back({ std::move(leg), leg_start });
        } else {
            right.push_back({ std::move(leg), leg_start });
        }
    }

    // size_t index = 0;
    // for (const auto &[left, right] : make_pairs(left, right, PyList_Size(pair_labels))) {
    //     auto left_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 0);
    //     auto right_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 1);

    //     for (const auto &point : left) {
    //         PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), left_label);
    //     }
    //     for (const auto &point : right) {
    //         PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), right_label);
    //     }
    //     index++;
    // }
}

// static std::vector<LegPair> make_pairs(LegWithStart &left, LegWithStart &right, Py_ssize_t size)
// {
//     std::vector<LegPair> pairs;

//     bool left_full = left.size() == size;
//     bool right_full = right.size() == size;

//     auto l = left.begin();
//     auto r = right.begin();

//     while ((!left.empty() || !right.empty()) && pairs.size() < size) {
//         if (left.empty()) {
//             pairs.emplace_back(std::vector<Point>(), std::move(r->first));
//             r++;
//         } else if (right.empty()) {
//             pairs.emplace_back(std::move(l->first), std::vector<Point>());
//             l++;
//         } else {
//             if ((left_full && right_full) || ((r + 1 == right.end() || is_closer(l, r)) && (l + 1 != left.end() || is_closer(r, l)))) {
//                 //  ((r + 1 == right.end() || abs(l->second - r->second) < abs(l->second - (r + 1)->second)) && (l + 1 == left.end() || abs(l->second - r->second) < abs((l + 1)->second - r->second)))) {
//                 pairs.emplace_back(std::move(l->first), std::move(r->first));
//                 l++;
//                 r++;
//             }
//         }
//         else if (left_full || (l->second < r->second && !right_full))
//         {
//             pairs.emplace_back(std::move(l->first), std::vector<Point>());
//             l++;
//         }
//         else
//         {
//             pairs.emplace_back(std::vector<Point>(), std::move(r->first));
//             r++;
//         }
//     }

//     return pairs;
// }

static bool is_closer(LegWithStart::iterator left, LegWithStart::iterator right)
{
    return Point::distance(left->second, right->second) < Point::distance(left->second, (right + 1)->second);
}
