#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// Longitude, ranges from -180 to 180, negative values are to the right of the zero meridian when the globe is viewed with the north pole at the top
    long: f64,
    /// Latitude, ranges from -90 (south pole) to 90 (north pole)
    lat: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectedPoint<'p, P: Projection> {
    pub x: f64,
    pub y: f64,
    projection: &'p P,
}

pub trait Projectable<P: Projection> {
    fn project<'p>(&self, projection: &'p P) -> ProjectedPoint<'p, P>;
}

pub trait Projection {
    fn project(&self, point: &Point) -> (f64, f64);
    fn invert(&self, projected_point: (f64, f64)) -> Point;
}

#[derive(Clone, Copy, Debug)]
pub struct Equirectangular {
    central_long: f64,
    true_scale_lat: f64,
}

pub struct EquirectangularBuilder {
    _inner: Equirectangular,
}

impl EquirectangularBuilder {
    pub fn central_long(mut self, long: f64) -> Self {
        self._inner.central_long = long;
        self
    }

    pub fn true_scale_lat(mut self, lat: f64) -> Self {
        self._inner.true_scale_lat = lat;
        self
    }

    pub fn build(self) -> Equirectangular {
        self._inner
    }
}

/// The projected points from this projection form a rectangle whose y-axis always runs from -1 to 1
/// and whose x-axis runs from -s to s, where s is 2*cos(true_scale_lat), twice the cosine of the latitude at
/// which the projection is true.
impl Equirectangular {
    pub fn new() -> EquirectangularBuilder {
        EquirectangularBuilder {
            _inner: Equirectangular {
                central_long: 0.,
                true_scale_lat: 0.,
            },
        }
    }
}

impl Projection for Equirectangular {
    fn project(&self, point: &Point) -> (f64, f64) {
        let x = (point.long - self.central_long) / 180.;
        let y = (point.lat) / 90.;
        (
            if x < -1. {
                x + 2.
            } else {
                if x > 1. {
                    x - 2.
                } else {
                    x
                }
            } * 2.
                * self.true_scale_lat.to_radians().cos(),
            y,
        )
    }

    fn invert(&self, projected_point: (f64, f64)) -> Point {
        let long = 180. * projected_point.0 / (2. * self.true_scale_lat.to_radians().cos())
            + self.central_long;
        let lat = 90. * projected_point.1;
        Point {
            long: if long > 180. {
                long - 360.
            } else {
                if long < -180. {
                    long + 360.
                } else {
                    long
                }
            },
            lat,
        }
    }
}

impl Point {
    pub fn new(long: f64, lat: f64) -> Self {
        Point { long, lat }
    }
}

impl<P: Projection> Projectable<P> for Point {
    fn project<'p>(&self, projection: &'p P) -> ProjectedPoint<'p, P> {
        let projected_point = projection.project(&self);
        ProjectedPoint {
            x: projected_point.0,
            y: projected_point.1,
            projection,
        }
    }
}

impl From<(f64, f64)> for Point {
    fn from(value: (f64, f64)) -> Self {
        Point {
            long: value.0,
            lat: value.1,
        }
    }
}

impl<'p, P: Projection> ProjectedPoint<'p, P> {
    pub fn point(&self) -> Point {
        self.projection.invert((self.x, self.y))
    }

    pub fn from_normalized((x, y): (f64, f64), projection: &'p P) -> Self {
        ProjectedPoint { x, y, projection }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    impl<'p, P: Projection> float_eq::FloatEq for ProjectedPoint<'p, P> {
        type Tol = f64;

        fn eq_abs(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.x.eq_abs(&other.x, tol) && self.y.eq_abs(&other.y, tol)
        }

        fn eq_rmax(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.x.eq_rmax(&other.x, tol) && self.y.eq_rmax(&other.y, tol)
        }

        fn eq_rmin(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.x.eq_rmin(&other.x, tol) && self.y.eq_rmin(&other.y, tol)
        }

        fn eq_r1st(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.x.eq_r1st(&other.x, tol) && self.y.eq_r1st(&other.y, tol)
        }

        fn eq_r2nd(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.x.eq_r2nd(&other.x, tol) && self.y.eq_r2nd(&other.y, tol)
        }

        fn eq_ulps(&self, other: &Self, tol: &float_eq::UlpsTol<Self::Tol>) -> bool {
            self.x.eq_ulps(&other.x, tol) && self.y.eq_ulps(&other.y, tol)
        }
    }

    impl<'p, P: Projection> float_eq::AssertFloatEq for ProjectedPoint<'p, P> {
        type DebugAbsDiff = (f64, f64);

        type DebugTol = (f64, f64);

        fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
            let x_diff = self.x.debug_abs_diff(&other.x);
            let y_diff = self.y.debug_abs_diff(&other.y);
            (x_diff, y_diff)
        }

        fn debug_ulps_diff(&self, other: &Self) -> float_eq::DebugUlpsDiff<Self::DebugAbsDiff> {
            let x_diff = self.x.debug_ulps_diff(&other.x);
            let y_diff = self.y.debug_ulps_diff(&other.y);
            (x_diff, y_diff)
        }

        fn debug_abs_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.x.debug_abs_tol(&other.x, tol);
            let y_diff = self.y.debug_abs_tol(&other.y, tol);
            (x_diff, y_diff)
        }

        fn debug_rmax_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.x.debug_rmax_tol(&other.x, tol);
            let y_diff = self.y.debug_rmax_tol(&other.y, tol);
            (x_diff, y_diff)
        }

        fn debug_rmin_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.x.debug_rmin_tol(&other.x, tol);
            let y_diff = self.y.debug_rmin_tol(&other.y, tol);
            (x_diff, y_diff)
        }

        fn debug_r1st_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.x.debug_r1st_tol(&other.x, tol);
            let y_diff = self.y.debug_r1st_tol(&other.y, tol);
            (x_diff, y_diff)
        }

        fn debug_r2nd_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.x.debug_r2nd_tol(&other.x, tol);
            let y_diff = self.y.debug_r2nd_tol(&other.y, tol);
            (x_diff, y_diff)
        }

        fn debug_ulps_tol(
            &self,
            other: &Self,
            tol: &float_eq::UlpsTol<Self::Tol>,
        ) -> float_eq::UlpsTol<Self::DebugTol>
        where
            float_eq::UlpsTol<Self::DebugTol>: Sized,
        {
            let x_diff = self.x.debug_ulps_tol(&other.x, tol);
            let y_diff = self.y.debug_ulps_tol(&other.y, tol);
            (x_diff, y_diff)
        }
    }

    impl float_eq::FloatEq for Point {
        type Tol = f64;

        fn eq_abs(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.lat.eq_abs(&other.lat, tol) && self.long.eq_abs(&other.long, tol)
        }

        fn eq_rmax(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.lat.eq_rmax(&other.lat, tol) && self.long.eq_rmax(&other.long, tol)
        }

        fn eq_rmin(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.lat.eq_rmin(&other.lat, tol) && self.long.eq_rmin(&other.long, tol)
        }

        fn eq_r1st(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.lat.eq_r1st(&other.lat, tol) && self.long.eq_r1st(&other.long, tol)
        }

        fn eq_r2nd(&self, other: &Self, tol: &Self::Tol) -> bool {
            self.lat.eq_r2nd(&other.lat, tol) && self.long.eq_r2nd(&other.long, tol)
        }

        fn eq_ulps(&self, other: &Self, tol: &float_eq::UlpsTol<Self::Tol>) -> bool {
            self.lat.eq_ulps(&other.lat, tol) && self.long.eq_ulps(&other.long, tol)
        }
    }

    impl float_eq::AssertFloatEq for Point {
        type DebugAbsDiff = (f64, f64);

        type DebugTol = (f64, f64);

        fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
            let x_diff = self.long.debug_abs_diff(&other.long);
            let y_diff = self.lat.debug_abs_diff(&other.lat);
            (x_diff, y_diff)
        }

        fn debug_ulps_diff(&self, other: &Self) -> float_eq::DebugUlpsDiff<Self::DebugAbsDiff> {
            let x_diff = self.long.debug_ulps_diff(&other.long);
            let y_diff = self.lat.debug_ulps_diff(&other.lat);
            (x_diff, y_diff)
        }

        fn debug_abs_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.long.debug_abs_tol(&other.long, tol);
            let y_diff = self.lat.debug_abs_tol(&other.lat, tol);
            (x_diff, y_diff)
        }

        fn debug_rmax_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.long.debug_rmax_tol(&other.long, tol);
            let y_diff = self.lat.debug_rmax_tol(&other.lat, tol);
            (x_diff, y_diff)
        }

        fn debug_rmin_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.long.debug_rmin_tol(&other.long, tol);
            let y_diff = self.lat.debug_rmin_tol(&other.lat, tol);
            (x_diff, y_diff)
        }

        fn debug_r1st_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.long.debug_r1st_tol(&other.long, tol);
            let y_diff = self.lat.debug_r1st_tol(&other.lat, tol);
            (x_diff, y_diff)
        }

        fn debug_r2nd_tol(&self, other: &Self, tol: &Self::Tol) -> Self::DebugTol {
            let x_diff = self.long.debug_r2nd_tol(&other.long, tol);
            let y_diff = self.lat.debug_r2nd_tol(&other.lat, tol);
            (x_diff, y_diff)
        }

        fn debug_ulps_tol(
            &self,
            other: &Self,
            tol: &float_eq::UlpsTol<Self::Tol>,
        ) -> float_eq::UlpsTol<Self::DebugTol>
        where
            float_eq::UlpsTol<Self::DebugTol>: Sized,
        {
            let x_diff = self.long.debug_ulps_tol(&other.long, tol);
            let y_diff = self.lat.debug_ulps_tol(&other.lat, tol);
            (x_diff, y_diff)
        }
    }

    fn assert_projections<P: Projection + std::fmt::Debug>(
        input: Vec<(impl Into<Point>, (f64, f64))>,
        projection: P,
    ) {
        for (point, expected_projection) in input {
            let point = point.into();
            let actual_projection = point.project(&projection);
            let expected_projection =
                ProjectedPoint::from_normalized(expected_projection, &projection);
            assert_float_eq!(actual_projection, expected_projection, r2nd <= f64::EPSILON);
            assert_float_eq!(point, expected_projection.point(), r2nd <= f64::EPSILON);
        }
    }

    mod equirectangular {
        use super::*;
        #[test]
        fn default() {
            assert_projections(
                vec![
                    ((0., 0.), (0., 0.)),
                    ((-180., 0.), (-2., 0.)),
                    ((0., 90.), (0., 1.)),
                    ((30., 60.), (2. / 6., 2. / 3.)),
                ],
                Equirectangular {
                    central_long: 0.,
                    true_scale_lat: 0.,
                },
            );
        }

        #[test]
        fn from_the_other_side_of_the_world() {
            assert_projections(
                vec![
                    ((0., 0.), (-2., 0.)),
                    ((180., 0.), (0., 0.)),
                    ((90., 0.), (-1., 0.)),
                    ((-90., 0.), (1., 0.)),
                ],
                Equirectangular {
                    central_long: 180.,
                    true_scale_lat: 0.,
                },
            );
        }

        #[test]
        fn from_quarter_way_across_the_world() {
            assert_projections(
                vec![
                    ((0., 0.), (1., 0.)),
                    ((90., 0.), (2., 0.)),
                    ((135., 0.), (-1.5, 0.)),
                ],
                Equirectangular {
                    central_long: -90.,
                    true_scale_lat: 0.,
                },
            );
        }

        #[test]
        fn true_scale_not_equator() {
            assert_projections(
                vec![((0., 0.), (0., 0.)), ((180., 0.), (1., 0.))],
                Equirectangular {
                    central_long: 0.,
                    true_scale_lat: 60.,
                },
            );
        }
    }
}
