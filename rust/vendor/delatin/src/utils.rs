use crate::Point;

// TODO: consider `robust` library instead
pub(crate) fn get_signed_area(point_a: Point, point_b: Point, point_c: Point) -> i64 {
    let r1 = point_b.0 as i64 - point_c.0 as i64;
    let r2 = point_a.1 as i64 - point_c.1 as i64;
    let r3 = point_b.1 as i64 - point_c.1 as i64;
    let r4 = point_a.0 as i64 - point_c.0 as i64;

    r1 * r2 - r3 * r4
}

pub(crate) fn is_point_in_circumcircle(
    test_point: Point,
    point_a: Point,
    point_b: Point,
    point_c: Point,
) -> bool {
    let delta_x_a = point_a.0 as i64 - test_point.0 as i64;
    let delta_y_a = point_a.1 as i64 - test_point.1 as i64;
    let delta_x_b = point_b.0 as i64 - test_point.0 as i64;
    let delta_y_b = point_b.1 as i64 - test_point.1 as i64;
    let delta_x_c = point_c.0 as i64 - test_point.0 as i64;
    let delta_y_c = point_c.1 as i64 - test_point.1 as i64;

    let square_distance_a = delta_x_a * delta_x_a + delta_y_a * delta_y_a;
    let square_distance_b = delta_x_b * delta_x_b + delta_y_b * delta_y_b;
    let square_distance_c = delta_x_c * delta_x_c + delta_y_c * delta_y_c;

    delta_x_a * (delta_y_b * square_distance_c - square_distance_b * delta_y_c)
        - delta_y_a * (delta_x_b * square_distance_c - square_distance_b * delta_x_c)
        + square_distance_a * (delta_x_b * delta_y_c - delta_y_b * delta_x_c)
        < 0
}
