# delatin 0.2.1, vendored

From crates.io (https://github.com/pokhanto/delatin-rs), ISC license, by its authors.

One change: the integer geometry (`get_signed_area`, `is_point_in_circumcircle`, the triangle
rasterization) uses `i64` instead of `isize`. On wasm32 `isize` is 32 bits, and the in-circle
determinant (fourth powers of cell offsets) overflows silently once a grid is about 900 cells
across; the triangulation then flips edges forever. Natively the results are unchanged.
