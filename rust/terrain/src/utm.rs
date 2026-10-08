//! Universal Transverse Mercator on the WGS 84 ellipsoid (GRS 80 differs by 0.1 mm), by Krüger's
//! series to sixth order in n (Karney 2011, "Transverse Mercator with an accuracy of a few
//! nanometers"): millimeter accuracy well beyond a zone's width. No datum shift: NAD83 latitude
//! and longitude go to NAD83 UTM, WGS 84 to WGS 84 UTM.

const A_AXIS: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const K0: f64 = 0.9996;
const E0: f64 = 500_000.0;
const N0_SOUTH: f64 = 10_000_000.0;

struct Series {
    n: f64,
    big_a: f64,
    alpha: [f64; 6],
    beta: [f64; 6],
    delta: [f64; 6],
}

fn series() -> Series {
    let n = F / (2.0 - F);
    let (n2, n3, n4, n5, n6) = (n * n, n.powi(3), n.powi(4), n.powi(5), n.powi(6));
    let big_a = A_AXIS / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0);
    let alpha = [
        n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0 + 7891.0 * n6 / 37800.0,
        13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0 + 281.0 * n5 / 630.0 - 1983433.0 * n6 / 1935360.0,
        61.0 * n3 / 240.0 - 103.0 * n4 / 140.0 + 15061.0 * n5 / 26880.0 + 167603.0 * n6 / 181440.0,
        49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0 + 6601661.0 * n6 / 7257600.0,
        34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,
        212378941.0 * n6 / 319334400.0,
    ];
    let beta = [
        n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0 - n4 / 360.0 - 81.0 * n5 / 512.0 + 96199.0 * n6 / 604800.0,
        n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1440.0 + 46.0 * n5 / 105.0 - 1118711.0 * n6 / 3870720.0,
        17.0 * n3 / 480.0 - 37.0 * n4 / 840.0 - 209.0 * n5 / 4480.0 + 5569.0 * n6 / 90720.0,
        4397.0 * n4 / 161280.0 - 11.0 * n5 / 504.0 - 830251.0 * n6 / 7257600.0,
        4583.0 * n5 / 161280.0 - 108847.0 * n6 / 3991680.0,
        20648693.0 * n6 / 638668800.0,
    ];
    let delta = [
        2.0 * n - 2.0 * n2 / 3.0 - 2.0 * n3 + 116.0 * n4 / 45.0 + 26.0 * n5 / 45.0 - 2854.0 * n6 / 675.0,
        7.0 * n2 / 3.0 - 8.0 * n3 / 5.0 - 227.0 * n4 / 45.0 + 2704.0 * n5 / 315.0 + 2323.0 * n6 / 945.0,
        56.0 * n3 / 15.0 - 136.0 * n4 / 35.0 - 1262.0 * n5 / 105.0 + 73814.0 * n6 / 2835.0,
        4279.0 * n4 / 630.0 - 332.0 * n5 / 35.0 - 399572.0 * n6 / 14175.0,
        4174.0 * n5 / 315.0 - 144838.0 * n6 / 6237.0,
        601676.0 * n6 / 22275.0,
    ];
    Series { n, big_a, alpha, beta, delta }
}

/// The UTM zone (1 to 60) containing a longitude (degrees).
pub fn zone_for(lon: f64) -> u32 {
    ((((lon + 180.0) / 6.0).floor() as i64).rem_euclid(60) + 1) as u32
}

fn central_meridian(zone: u32) -> f64 {
    (zone as f64 * 6.0 - 183.0).to_radians()
}

/// (easting, northing) in meters of (lon, lat) in degrees, in a zone; `north` for the northern
/// hemisphere's false northing.
pub fn to_utm(lon: f64, lat: f64, zone: u32, north: bool) -> (f64, f64) {
    let s = series();
    let (phi, lam) = (lat.to_radians(), lon.to_radians() - central_meridian(zone));
    let c = 2.0 * s.n.sqrt() / (1.0 + s.n);
    let t = (phi.sin().atanh() - c * (c * phi.sin()).atanh()).sinh();
    let xi_p = t.atan2(lam.cos());
    let eta_p = (lam.sin() / (1.0 + t * t).sqrt()).atanh();
    let (mut xi, mut eta) = (xi_p, eta_p);
    for (j, a) in s.alpha.iter().enumerate() {
        let k = 2.0 * (j + 1) as f64;
        xi += a * (k * xi_p).sin() * (k * eta_p).cosh();
        eta += a * (k * xi_p).cos() * (k * eta_p).sinh();
    }
    let n0 = if north { 0.0 } else { N0_SOUTH };
    (E0 + K0 * s.big_a * eta, n0 + K0 * s.big_a * xi)
}

/// (lon, lat) in degrees of a UTM position.
pub fn from_utm(e: f64, n: f64, zone: u32, north: bool) -> (f64, f64) {
    let s = series();
    let n0 = if north { 0.0 } else { N0_SOUTH };
    let xi = (n - n0) / (K0 * s.big_a);
    let eta = (e - E0) / (K0 * s.big_a);
    let (mut xi_p, mut eta_p) = (xi, eta);
    for (j, b) in s.beta.iter().enumerate() {
        let k = 2.0 * (j + 1) as f64;
        xi_p -= b * (k * xi).sin() * (k * eta).cosh();
        eta_p -= b * (k * xi).cos() * (k * eta).sinh();
    }
    let chi = (xi_p.sin() / eta_p.cosh()).asin();
    let mut phi = chi;
    for (j, d) in s.delta.iter().enumerate() {
        phi += d * (2.0 * (j + 1) as f64 * chi).sin();
    }
    let lam = central_meridian(zone) + eta_p.sinh().atan2(xi_p.cos());
    (lam.to_degrees(), phi.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_points_and_round_trip() {
        // The central meridian of zone 11 at the equator: (500000, 0).
        let (e, n) = to_utm(-117.0, 0.0, 11, true);
        assert!((e - 500_000.0).abs() < 1e-6 && n.abs() < 1e-6);
        // Round trips across a zone, to well under a millimeter.
        for &(lon, lat) in &[(-115.5, 44.95), (-119.9, 30.0), (-114.1, 60.0), (-117.0, -33.0)] {
            let north = lat >= 0.0;
            let (e, n) = to_utm(lon, lat, 11, north);
            let (lo, la) = from_utm(e, n, 11, north);
            assert!((lo - lon).abs() < 1e-9 && (la - lat).abs() < 1e-9, "{lon},{lat} -> {lo},{la}");
        }
        // Against pyproj (EPSG:4326 to 32611 / 32711), to a millimeter.
        for &(lon, lat, north, e0, n0) in &[
            (-115.5, 44.95, true, 618325.7667, 4978490.4521),
            (-119.9, 30.0, true, 220241.5549, 3322327.2251),
            (-114.1, 60.0, true, 661720.7509, 6654956.7199),
            (-117.0, -33.0, false, 500000.0000, 6348713.0560),
        ] {
            let (e, n) = to_utm(lon, lat, 11, north);
            assert!((e - e0).abs() < 1e-3 && (n - n0).abs() < 1e-3, "{lon},{lat}: {e},{n} vs {e0},{n0}");
        }
        assert_eq!(zone_for(-115.5), 11);
        assert_eq!(zone_for(179.9), 60);
        assert_eq!(zone_for(-180.0), 1);
    }
}
