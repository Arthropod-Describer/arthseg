#include <vector>

#include "types.hpp"

class Moments
{
  private:
    double theta;
    double radius;

  public:
    Moments(const std::vector<Point> &points)
    {
        auto centroid = get_centroid(points);
        int central_moment_11 = 0;
        int central_moment_20 = 0;
        int central_moment_02 = 0;

        for (const auto &[row, col] : points) {
            central_moment_11 += (row - centroid.row) * (col - centroid.col);
            central_moment_20 += pow(row - centroid.row, 2);
            central_moment_02 += pow(col - centroid.col, 2);
        }

        theta = 0.5 * atan2(2 * central_moment_11, central_moment_20 - central_moment_02);
        radius = centroid.row * sin(theta) + centroid.col * cos(theta);
    }

    Point project(const Point &point) const
    {
        return {
            (size_t) (point.row * sin(theta) + point.col * cos(theta) - radius),
            (size_t) (point.row * cos(theta) - point.col * sin(theta))
        };
    }

    int half_axis(const Point &point) const
    /**
     * @param point Point to calculate the half axis for
     * @return negative half axis if point is on the left side of the centroid, positive half axis otherwise
     */
    {
        return point.row * sin(theta) + point.col * cos(theta) - radius;
    }

    static Point get_centroid(const std::vector<Point> &points)
    {
        size_t row = 0, col = 0;
        for (const auto &point : points) {
            row += point.row;
            col += point.col;
        }
        return { row /= points.size(), col /= points.size() };
    }
};
