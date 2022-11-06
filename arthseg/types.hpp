#pragma once

#include <tuple>

struct Point
{
    size_t row, col;
    Point(size_t row, size_t col) : row(row), col(col) {}
    static float distance(const Point &a, const Point &b)
    {
        float dx = abs((int) a.col - (int) b.col);
        float dy = abs((int) a.row - (int) b.row);
        return dx > dy ? 0.41 * dy + 0.941246 * dx : 0.41 * dx + 0.941246 * dy;
    }
};

template <typename T>
class Matrix
{
  private:
    std::vector<T> data;

  public:
    size_t rows, cols;

    Matrix(const size_t rows, const size_t cols) : data(rows * cols), rows(rows), cols(cols) {}
    T &at(const size_t row, const size_t col) { return data[row * cols + col]; }
    T &at(const Point &point) { return data[point.row * cols + point.col]; }
    const T &at(const size_t row, const size_t col) const { return data[row * cols + col]; }
    const T &at(const Point &point) const { return data[point.row * cols + point.col]; }
};
