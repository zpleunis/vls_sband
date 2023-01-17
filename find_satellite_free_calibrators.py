"""Select calibrators in the satellite-free zone for VLA S-band observatoins.

It is a bit of a clunky script, but sometimes you just have to get your
observations set up quickly.

"""


import astropy.units as u
import numpy as np

from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time


def equatorial2horizon(ha, dec, lat):

    altitude = np.rad2deg(np.arcsin(
        np.sin(np.deg2rad(dec)) * np.sin(np.deg2rad(lat)) \
        + np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(lat)) \
        * np.cos(np.deg2rad(ha))
    ))

    azimuth = np.rad2deg(np.arccos(
        (np.sin(np.deg2rad(dec)) - np.sin(np.deg2rad(lat)) \
        * np.sin(np.deg2rad(altitude))) \
        / (np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(altitude)))
    ))

    return altitude, azimuth


def angular_separation(ra1, dec1, ra2, dec2):
    """Calculates the angular separation between two celestial objects.
    Parameters
    ----------
    ra1 : float, array
        Right ascension of the first source in degrees.
    dec1 : float, array
        Declination of the first source in degrees.
    ra2 : float, array
        Right ascension of the second source in degrees.
    dec2 : float, array
        Declination of the second source in degrees.
    Returns
    -------
    angle : float, array
        Angle between the two sources in degrees, where 0 corresponds
        to the positive Dec axis.
    angular_separation : float, array
        Angular separation between the two sources in degrees.
    Notes
    -----
    The angle between sources is calculated with the Pythagorean theorem
    and is used later to calculate the uncertainty in the event position
    in the direction of the known source.
    Calculating the angular separation using spherical geometry gives
    poor accuracy for small (< 1 degree) separations, and using the
    Pythagorean theorem fails for large separations (> 1 degrees).
    Transforming the spherical geometry cosine formula into one that
    uses haversines gives the best results, see e.g. [1]_. This gives:
    .. math:: \mathrm{hav} d = \mathrm{hav} \Delta\delta +
              \cos\delta_1 \cos\delta_2 \mathrm{hav} \Delta\\alpha
    Where we use the identity
    :math:`\mathrm{hav} \\theta = \sin^2(\\theta/2)` in our
    calculations.
    The calculation might give inaccurate results for antipodal
    coordinates, but we do not expect such calculations here..
    The source angle (or bearing angle) :math:`\\theta` from a point
    A(ra1, dec1) to a point B(ra2, dec2), defined as the angle in
    clockwise direction from the positive declination axis can be
    calculated using:
    .. math:: \\tan(\\theta) = (\\alpha_2 - \\alpha_1) /
              (\delta_2 - \delta_1)
    In NumPy :math:`\\theta` can be calculated using the arctan2
    function. Note that for negative :math:`\\theta` a factor :math:`2\pi`
    needs to be added. See also the documentation for arctan2.
    References
    ----------
    .. [1] Sinnott, R. W. 1984, Sky and Telescope, 68, 158
    Examples
    --------
    >>> print(angular_separation(200.478971, 55.185900, 200.806433, 55.247994))
    (79.262937451490941, 0.19685681276638525)
    >>> print(angular_separation(0., 20., 180., 20.))
    (90.0, 140.0)
    """
    # convert decimal degrees to radians
    deg2rad = np.pi / 180
    ra1 = ra1 * deg2rad
    dec1 = dec1 * deg2rad
    ra2 = ra2 * deg2rad
    dec2 = dec2 * deg2rad

    # delta works
    dra = ra1 - ra2
    ddec = dec1 - dec2

    # haversine formula
    hav = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    angular_separation = 2 * np.arcsin(np.sqrt(hav))

    # angle in the clockwise direction from the positive dec axis
    # note the minus signs in front of `dra` and `ddec`
    source_angle = np.arctan2(-dra, -ddec)
    source_angle[source_angle < 0] += 2 * np.pi

    # convert radians back to decimal degrees
    return source_angle / deg2rad, angular_separation / deg2rad


def read_gain_calibrators(fname="gain_calibrators.txt"):
    """Read txt file with VLA gain calibrators."""

    cal_name = []
    cal_ras = []
    cal_decs = []

    with open(fname, "r") as f:
        for line in f:
            if "J2000" in line:
                cal_name.append(line.split()[0])
                cal_ras.append(line.split()[3])
                cal_decs.append(line.split()[4])

    cal_name = np.array(cal_name)
    cal_skypos = SkyCoord(cal_ras, cal_decs)

    return cal_name, cal_skypos


def find_cal_in_sfz(targets, cal_name, cal_skypos):

    vla_lat = 34.079 * u.deg
    vla_lon = -107.62 * u.deg

    all_lsts = Angle(np.linspace(0, 24, 24 * 6) * u.hour)

    for target in targets:

        h = all_lsts.deg - target.ra.deg

        altitude = np.rad2deg(np.arcsin(
            np.sin(np.deg2rad(vla_lat)) * np.sin(np.deg2rad(target.dec.deg)) \
            + np.cos(np.deg2rad(vla_lat)) * np.cos(np.deg2rad(target.dec.deg)) \
            * np.cos(np.deg2rad(h))))

        lst_window = altitude > 20. * u.deg

        print("target:", target)
        if (lst_window == True).all():
            lst_start_idx = 0
            lst_stop_idx = 24 * 6 - 1
            print("LST window:", all_lsts[lst_start_idx], all_lsts[lst_stop_idx])
        else:
            # find LST start and stop times
            lst_start_idx = np.argwhere(np.diff(lst_window.astype(int)) == 1)[0][0] + 1
            lst_stop_idx = np.argwhere(np.diff(lst_window.astype(int)) == -1)[0][0]
            print("LST window:", all_lsts[lst_start_idx], all_lsts[lst_stop_idx])

        if lst_stop_idx < lst_start_idx:
            lsts = np.roll(all_lsts[lst_window], -(lst_stop_idx + 1))
        else:
            lsts = all_lsts[lst_window]

        sfz_indices = []
        in_sfzs = []
        separations = []

        for i in range(len(cal_skypos)):

            h = lsts.deg - cal_skypos[i].ra.deg

            altitude, azimuth = equatorial2horizon(h, cal_skypos[i].dec.deg, vla_lat)

            in_sfz = ((250 * u.deg < azimuth) \
                    & (azimuth < 330 * u.deg) \
                    & (altitude > 30 * u.deg)) \
                    | ((45 * u.deg < azimuth) \
                    & (azimuth < 90 * u.deg) \
                    & (altitude < 60 * u.deg))

            # 3C286
            next_source = SkyCoord("03h31m08.28811s 30d30\'32.9600\"")

            # distance to next source
            ang, sep = angular_separation(
                np.array([next_source.ra.deg]), np.array([next_source.dec.deg]),
                np.array([cal_skypos[i].ra.deg]), np.array([cal_skypos[i].dec.deg]))

            in_sfzs.append(in_sfz)
            separations.append(sep[0])

        in_sfzs = np.array(in_sfzs)
        separations = np.array(separations)

        hour_indices = list(range(0, len(lsts), 6)) + [len(lsts) - 1]

        for j in range(len(hour_indices) - 1):

            sfz_mask = np.sum(in_sfzs[:,hour_indices[j]:hour_indices[j+1]],
                axis=1) == hour_indices[j+1] - hour_indices[j]

            best_cal_idx = np.argmin(separations[sfz_mask])

            print("{:.1f}-{:.1f} use {} at {}, {} with separation {:.1f} deg".format(
                lsts[hour_indices[j]], lsts[hour_indices[j+1]],
                cal_name[sfz_mask][best_cal_idx],
                cal_skypos[sfz_mask][best_cal_idx].ra.to_string(),
                cal_skypos[sfz_mask][best_cal_idx].dec.to_string(),
                separations[sfz_mask][best_cal_idx]))

        print()


if __name__ == "__main__":

    targets = [
        SkyCoord("22h38m49s -30d10\'50\""),
        SkyCoord("10h35m22.272s 73d45\'14.04\""),
        SkyCoord("22h06m37.416s 17d21\'45.72\""),
        SkyCoord("18h25m29.136s 81d25\'06.6\""),
        SkyCoord("22h16m04.77s -07d53\'53.7\""),
        SkyCoord("01h33m47s 31d51\'30\""),
        SkyCoord("09h57m54.6994s 68d49\'00.8529\"")
    ]

    cal_name, cal_skypos = read_gain_calibrators()
    find_cal_in_sfz(targets, cal_name, cal_skypos)
