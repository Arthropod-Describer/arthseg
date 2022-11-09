#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <algorithm>
#include <numpy/arrayobject.h>

#include "legs.hpp"
#include "moments.hpp"

#include <iostream>

using LegWithHeight = std::vector<std::pair<std::vector<Point>, size_t>>;
using LegPair = std::pair<std::vector<Point>, std::vector<Point>>;

static std::vector<LegPair> make_pairs(LegWithHeight &left, LegWithHeight &right, size_t size);
static bool is_closer(LegWithHeight::iterator left, LegWithHeight::iterator right);

void reored_legs(PyArrayObject *image, PyObject *body_labels, PyObject *pair_labels, const std::vector<std::vector<Point>> &legs, const std::vector<Point> &body)
{
    LegWithHeight left, right;

    auto body_moments = Moments(body);
    // for (npy_intp row = 0; row < PyArray_DIM(image, 0); row++) {
    //     // for (npy_intp col = 0; col < PyArray_DIM(image, 1); col++) {
    //     auto col = (body_moments.radius - row * sin(body_moments.angle)) / cos(body_moments.angle);

    //     if (col < 0 || col >= PyArray_DIM(image, 1)) {
    //         continue;
    //     }
    //     PyArray_Set(image, row, col, 10);
    //     // }
    // }
    Point x_axis_intersection(0, body_moments.radius / cos(body_moments.angle));

    for (const auto &leg : legs) {
        auto centroid = Moments::get_centroid(leg);
        auto leg_start = Moments::get_centroid(find_leg_start(image, body_labels, leg));

        if (body_moments.half_axis(centroid) < 0) {
            left.push_back({ std::move(leg), Point::distance(x_axis_intersection, body_moments.project(leg_start)) });
        } else {
            right.push_back({ std::move(leg), Point::distance(x_axis_intersection, body_moments.project(leg_start)) });
        }
    }

    std::sort(left.begin(), left.end(), [](auto &a, auto &b) { return a.second < b.second; });
    std::sort(right.begin(), right.end(), [](auto &a, auto &b) { return a.second < b.second; });

    size_t index = 0;
    for (const auto &[left, right] : make_pairs(left, right, PyList_Size(pair_labels))) {
        auto left_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 0);
        auto right_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 1);

        for (const auto &point : left) {
            PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), left_label);
        }
        for (const auto &point : right) {
            PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), right_label);
        }
        index++;
    }
}

static std::vector<LegPair> make_pairs(LegWithHeight &left, LegWithHeight &right, size_t size)
{
    std::vector<LegPair> pairs;

    bool left_full = left.size() == size;
    bool right_full = right.size() == size;

    auto l = left.begin();
    auto r = right.begin();

    while ((!left.empty() || !right.empty()) && pairs.size() < size) {
        if (left.empty()) {
            pairs.emplace_back(std::vector<Point>(), std::move(r->first));
            r++;
        } else if (right.empty()) {
            pairs.emplace_back(std::move(l->first), std::vector<Point>());
            l++;
        } else {
            if ((left_full && right_full) || ((r + 1 == right.end() || is_closer(l, r)) && (l + 1 != left.end() || is_closer(r, l)))) {
                //  ((r + 1 == right.end() || abs(l->second - r->second) < abs(l->second - (r + 1)->second)) && (l + 1 == left.end() || abs(l->second - r->second) < abs((l + 1)->second - r->second)))) {
                pairs.emplace_back(std::move(l->first), std::move(r->first));
                l++;
                r++;
            } else if (left_full || (l->second < r->second && !right_full)) {
                pairs.emplace_back(std::move(l->first), std::vector<Point>());
                l++;
            } else {
                pairs.emplace_back(std::vector<Point>(), std::move(r->first));
                r++;
            }
        }
    }

    return pairs;
}

static bool is_closer(LegWithHeight::iterator left, LegWithHeight::iterator right)
{
    return abs((int) left->second - (int) right->second) < abs((int) left->second - (int) (right + 1)->second);
}
