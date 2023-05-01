use image::{DynamicImage, GenericImageView, ImageBuffer};

pub struct Map<P: Projection> {
    image: DynamicImage,
    projection: P,
}

impl<P: Projection> Map<P> {
    pub fn new(image: impl Into<DynamicImage>, projection: P) -> Self {
        Map {
            image: image.into(),
            projection,
        }
    }

    pub fn image(&self) -> &impl GenericImageView {
        &self.image
    }

    pub fn to_image(self) -> DynamicImage {
        self.image
    }

    pub fn projection(&self) -> &P {
        &self.projection
    }

    pub fn convert_to<Q: Projection>(self, projection: Q) -> Map<Q> {
        let image_dimensions = self.image.dimensions();
        let source_projection_dimensions = self.projection.dimensions();
        let target_projection_dimensions = projection.dimensions();
        let scale = (target_projection_dimensions.width.to
            - target_projection_dimensions.width.from)
            / (target_projection_dimensions.height.to - target_projection_dimensions.height.from);
        let new_image_dimensions = (
            (scale * image_dimensions.1 as f64) as u32,
            image_dimensions.1,
        );
        let (source_coordinate_increment_x, source_coordinate_increment_y) = (
            (source_projection_dimensions.width.to - source_projection_dimensions.width.from)
                / image_dimensions.0 as f64,
            (source_projection_dimensions.height.to - source_projection_dimensions.height.from)
                / image_dimensions.1 as f64,
        );
        let (target_coordinate_increment_x, target_coordinate_increment_y) = (
            (target_projection_dimensions.width.to - target_projection_dimensions.width.from)
                / new_image_dimensions.0 as f64,
            (target_projection_dimensions.height.to - target_projection_dimensions.height.from)
                / new_image_dimensions.1 as f64,
        );
        let mut new_image = ImageBuffer::new(new_image_dimensions.0, new_image_dimensions.1);
        for target_pixel_x in (0..new_image_dimensions.0).map(|i| i as f64) {
            for target_pixel_y in (0..new_image_dimensions.1).map(|i| i as f64) {
                let target_upper_left = (
                    target_projection_dimensions.width.from
                        + target_coordinate_increment_x * target_pixel_x,
                    target_projection_dimensions.height.from
                        + target_coordinate_increment_y * target_pixel_y,
                );
                let target_lower_right = (
                    target_upper_left.0 + target_coordinate_increment_x,
                    target_upper_left.1 + target_coordinate_increment_y,
                );
                let target_upper_left =
                    ProjectedPoint::from_normalized(target_upper_left, &projection);
                let target_lower_right =
                    ProjectedPoint::from_normalized(target_lower_right, &projection);
                if !target_upper_left.is_err() && !target_lower_right.is_err() {
                    let target_upper_left = target_upper_left.unwrap();
                    let target_lower_right = target_lower_right.unwrap();
                    let source_upper_left = target_upper_left.project(&self.projection);
                    let source_lower_right = target_lower_right.project(&self.projection);
                    let mut source_pixel_x = ((source_upper_left.x
                        - source_projection_dimensions.width.from)
                        / source_coordinate_increment_x)
                        .floor() as u32;
                    if source_pixel_x == image_dimensions.0 {
                        source_pixel_x = 0;
                    }
                    let mut source_pixel_y = ((source_upper_left.y
                        - source_projection_dimensions.height.from)
                        / source_coordinate_increment_y)
                        .floor() as u32;
                    if source_pixel_y == image_dimensions.1 {
                        source_pixel_y = 0;
                    }
                    new_image.put_pixel(
                        target_pixel_x as u32,
                        target_pixel_y as u32,
                        self.image.get_pixel(source_pixel_x, source_pixel_y),
                    );
                }
            }
        }
        Map {
            image: new_image.into(),
            projection,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// Longitude, ranges from -180 to 180, negative values are to the right of the zero meridian when the globe is viewed with the north pole at the top
    long: f64,
    /// Latitude, ranges from -90 (south pole) to 90 (north pole)
    lat: f64,
}

impl Point {
    pub fn long(&self) -> f64 {
        self.long
    }
    pub fn lat(&self) -> f64 {
        self.lat
    }
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

#[derive(Debug)]
pub struct Interval<T> {
    pub from: T,
    pub to: T,
}

impl<T: PartialOrd> Interval<T> {
    pub fn is_within(&self, value: &T) -> bool {
        self.from <= *value && *value <= self.to
    }
}

#[derive(Debug)]
pub struct Dimensions {
    width: Interval<f64>,
    height: Interval<f64>,
}

impl Dimensions {
    pub fn is_within(&self, point: (f64, f64)) -> bool {
        self.width.is_within(&point.0) && self.height.is_within(&point.1)
    }
}

pub trait Projection {
    fn dimensions(&self) -> Dimensions;
    fn project(&self, point: &Point) -> (f64, f64);
    fn invert(&self, projected_point: (f64, f64)) -> Point;
    fn convert_point<Q: Projection>(
        &self,
        projected_point: (f64, f64),
        target_projection: &Q,
    ) -> (f64, f64) {
        target_projection.project(&self.invert(projected_point))
    }
    fn projected_point_within_bounds(&self, point: (f64, f64)) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct AzimuthalEquidistant {
    center: Point,
    central_long: f64,
}

impl AzimuthalEquidistant {
    pub fn new() -> AzimuthalEquidistantBuilder {
        AzimuthalEquidistantBuilder {
            center: Point { lat: 90., long: 0. },
            central_long: 0.,
        }
    }
}

impl AzimuthalEquidistant {
    pub fn center(&self) -> Point {
        self.center
    }
    pub fn central_long(&self) -> f64 {
        self.central_long
    }
}

impl Projection for AzimuthalEquidistant {
    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: Interval { from: -1., to: 1. },
            height: Interval { from: -1., to: 1. },
        }
    }

    fn project(&self, point: &Point) -> (f64, f64) {
        // https://mathworld.wolfram.com/AzimuthalEquidistantProjection.html
        let angular_distance = (point.lat.to_radians().sin() * self.center.lat.to_radians().sin()
            + point.lat.to_radians().cos()
                * self.center.lat.to_radians().cos()
                * (point.long - self.center.long).to_radians().cos())
        .acos();
        let k = if angular_distance != 0. {
            angular_distance / angular_distance.sin()
        } else {
            0.
        };
        let x =
            k * point.lat.to_radians().cos() * (point.long - self.center.long).to_radians().sin()
                / std::f64::consts::PI;
        let y = k
            * (point.lat.to_radians().sin() * self.center.lat.to_radians().cos()
                - point.lat.to_radians().cos()
                    * self.center.lat.to_radians().sin()
                    * (point.long - self.center.long).to_radians().cos())
            / std::f64::consts::PI;
        if self.central_long != 0. {
            (
                x * self.central_long.to_radians().cos() + y * self.central_long.to_radians().sin(),
                -x * self.central_long.to_radians().sin()
                    + y * self.central_long.to_radians().cos(),
            )
        } else {
            (x, y)
        }
    }

    fn invert(&self, projected_point: (f64, f64)) -> Point {
        let mut projected_point = (
            projected_point.0 * std::f64::consts::PI,
            projected_point.1 * std::f64::consts::PI,
        );
        if self.central_long != 0. {
            projected_point = (
                projected_point.0 * self.central_long.to_radians().cos()
                    - projected_point.1 * self.central_long.to_radians().sin(),
                projected_point.0 * self.central_long.to_radians().sin()
                    + projected_point.1 * self.central_long.to_radians().cos(),
            )
        }
        let angular_distance =
            (projected_point.0 * projected_point.0 + projected_point.1 * projected_point.1).sqrt();
        if angular_distance == 0. {
            return self.center;
        }
        let lat = (angular_distance.cos() * self.center.lat.to_radians().sin()
            + projected_point.1 * angular_distance.sin() * self.center.lat.to_radians().cos()
                / angular_distance)
            .asin()
            .to_degrees();
        let long = self.center.long
            + (projected_point.0 * angular_distance.sin())
                .atan2(
                    angular_distance * self.center.lat.to_radians().cos() * angular_distance.cos()
                        - projected_point.1
                            * self.center.lat.to_radians().sin()
                            * angular_distance.sin(),
                )
                .to_degrees();
        Point { lat, long }
    }

    fn projected_point_within_bounds(&self, point: (f64, f64)) -> bool {
        if self.dimensions().is_within(point) {
            (point.0 * point.0 + point.1 * point.1) <= 1.
        } else {
            false
        }
    }
}

pub struct AzimuthalEquidistantBuilder {
    center: Point,
    central_long: f64,
}

impl AzimuthalEquidistantBuilder {
    pub fn center(mut self, center: Point) -> Self {
        self.center = center;
        self
    }

    pub fn central_long(mut self, central_long: f64) -> Self {
        self.central_long = central_long;
        self
    }

    pub fn build(self) -> AzimuthalEquidistant {
        AzimuthalEquidistant {
            center: self.center,
            central_long: self.central_long,
        }
    }
}

/// The projected points from this projection form a rectangle whose y-axis always runs from -1 to 1
/// and whose x-axis runs from -s to s, where s is 2 * cos(true_scale_lat), twice the cosine of the latitude at
/// which the projection is true.
#[derive(Clone, Copy, Debug)]
pub struct Equirectangular {
    central_long: f64,
    true_scale_lat: f64,
    width: f64,
}

impl Equirectangular {
    pub fn central_long(&self) -> f64 {
        self.central_long
    }
    pub fn true_scale_lat(&self) -> f64 {
        self.true_scale_lat
    }
}

pub struct EquirectangularBuilder {
    central_long: f64,
    true_scale_lat: f64,
}

impl EquirectangularBuilder {
    pub fn central_long(mut self, long: f64) -> Self {
        self.central_long = long;
        self
    }

    pub fn true_scale_lat(mut self, lat: f64) -> Self {
        self.true_scale_lat = lat;
        self
    }

    pub fn build(self) -> Equirectangular {
        Equirectangular {
            central_long: self.central_long,
            true_scale_lat: self.true_scale_lat,
            width: self.true_scale_lat.to_radians().cos(),
        }
    }
}

impl Equirectangular {
    pub fn new() -> EquirectangularBuilder {
        EquirectangularBuilder {
            central_long: 0.,
            true_scale_lat: 0.,
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
            } * (2. * self.width),
            y,
        )
    }

    fn invert(&self, projected_point: (f64, f64)) -> Point {
        let long = 180. * projected_point.0 / (2. * self.width) + self.central_long;
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

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: Interval {
                from: -2. * self.width,
                to: 2. * self.width,
            },
            height: Interval { from: -1., to: 1. },
        }
    }

    fn projected_point_within_bounds(&self, point: (f64, f64)) -> bool {
        self.dimensions().is_within(point)
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

impl TryFrom<(f64, f64)> for Point {
    type Error = ();

    fn try_from(value: (f64, f64)) -> Result<Self, Self::Error> {
        if -180. <= value.0 && value.0 <= 180. && -90. <= value.1 && value.1 <= 90. {
            Ok(Point {
                long: value.0,
                lat: value.1,
            })
        } else {
            Err(())
        }
    }
}

impl<'p, P: Projection> ProjectedPoint<'p, P> {
    pub fn point(&self) -> Point {
        self.projection.invert((self.x, self.y))
    }
    /// The error occurs when the point (x, y) is outside of the range the projection would normally produce
    pub fn from_normalized((x, y): (f64, f64), projection: &'p P) -> Result<Self, ()> {
        if projection.projected_point_within_bounds((x, y)) {
            Ok(ProjectedPoint { x, y, projection })
        } else {
            Err(())
        }
    }
}

impl<'p, P: Projection, Q: Projection> Projectable<Q> for ProjectedPoint<'p, P> {
    fn project<'q>(&self, projection: &'q Q) -> ProjectedPoint<'q, Q> {
        let (x, y) = self.projection.convert_point((self.x, self.y), projection);
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

    fn assert_projections<T: TryInto<Point> + std::fmt::Debug, P: Projection + std::fmt::Debug>(
        input: Vec<(T, (f64, f64))>,
        projection: P,
    ) where
        <T as TryInto<Point>>::Error: std::fmt::Debug,
    {
        for (point, expected_projection) in input {
            let point = point.try_into().unwrap();
            let actual_projection = point.project(&projection);
            let expected_projection =
                ProjectedPoint::from_normalized(expected_projection, &projection).unwrap();
            assert_float_eq!(actual_projection, expected_projection, abs <= 0.001);
            assert_float_eq!(point, expected_projection.point(), abs <= 0.001);
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
                Equirectangular::new()
                    .central_long(0.)
                    .true_scale_lat(0.)
                    .build(),
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
                Equirectangular::new()
                    .central_long(180.)
                    .true_scale_lat(0.)
                    .build(),
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
                Equirectangular::new()
                    .central_long(-90.)
                    .true_scale_lat(0.)
                    .build(),
            );
        }

        #[test]
        fn true_scale_not_equator() {
            assert_projections(
                vec![
                    ((0., 0.), (0., 0.)),
                    ((180., 0.), (1., 0.)),
                    ((-90., 0.), (-1. / 2., 0.)),
                ],
                Equirectangular::new()
                    .central_long(0.)
                    .true_scale_lat(60.)
                    .build(),
            );
        }

        #[test]
        fn different_central_longs() {
            let default_projection = Equirectangular::new().build();
            let shifted_projection = Equirectangular::new().central_long(90.).build();
            let projected_point =
                ProjectedPoint::from_normalized((0., 0.), &default_projection).unwrap();
            let converted_projected_point = projected_point.project(&shifted_projection);
            assert_float_eq!(
                converted_projected_point,
                ProjectedPoint::from_normalized((-1., 0.), &shifted_projection).unwrap(),
                r2nd <= f64::EPSILON
            );
        }

        #[test]
        fn different_true_scale_lats() {
            let default_projection = Equirectangular::new().build();
            let shifted_projection = Equirectangular::new().true_scale_lat(60.).build();
            let projected_point =
                ProjectedPoint::from_normalized((1., 0.), &default_projection).unwrap();
            let converted_projected_point = projected_point.project(&shifted_projection);
            assert_float_eq!(
                converted_projected_point,
                ProjectedPoint::from_normalized((0.5, 0.), &shifted_projection).unwrap(),
                r2nd <= f64::EPSILON
            );
        }
    }

    mod azimuthal_equidistant {
        use super::*;

        #[test]
        fn north_pole() {
            assert_projections(
                vec![
                    ((0., 90.), (0., 0.)),
                    ((90., 0.), (0.5, 0.)),
                    ((-90., 0.), (-0.5, 0.)),
                    ((0., 0.), (0., -0.5)),
                ],
                AzimuthalEquidistant::new()
                    .center((0., 90.).try_into().unwrap())
                    .build(),
            );
        }

        #[test]
        fn north_pole_shifted() {
            assert_projections(
                vec![
                    ((0., 90.), (0., 0.)),
                    ((90., 0.), (0., -0.5)),
                    ((-90., 0.), (0., 0.5)),
                    ((0., 0.), (-0.5, 0.)),
                ],
                AzimuthalEquidistant::new()
                    .center((0., 90.).try_into().unwrap())
                    .central_long(90.)
                    .build(),
            );
        }
    }
}
