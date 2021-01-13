from pyproj import Transformer
from geopy.distance import distance
import datetime
import numpy as np

TRANSFORMER = Transformer.from_crs("epsg:3857", "epsg:4326")


def geodesic_distance(o1, o2, is_3857=False):
    """
    Find the geodesic distance between two coordinates
    :param o1: coordinates of the first object in the form of (latitude, longitude)
    :param o2: coordinates of the second object in the form of (latitude, longitude)
    :param is_3857: boolean if coordinates is in EPSG 3857, otherwise it's EPSG 4326
    :return: geodesic distance in metres
    """
    if is_3857:
        lat1, long1 = TRANSFORMER.transform(o1[1], o1[0])
        lat2, long2 = TRANSFORMER. transform(o2[1], o2[0])
    else:
        lat1, long1 = o1[0], o1[1]
        lat2, long2 = o2[0], o2[1]

    return distance((lat1, long1), (lat2, long2)).meters


def chop_microseconds(delta):
    """
    Remove the microsecond part of timedelta, i.e. make microsecond 0
    :param delta: the timedelta object to be modified
    :return: a new timedelta object with the value modified from input
    """
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def time_difference(time1, time2):
    """
    Calculate the difference of second input from first input. Negative if second input is earlier than the first.
    :param time1: a time object
    :param time2: another time object
    :return: duration of time between time1 and time2
    """
    date = datetime.date(1, 1, 1)
    datetime1 = datetime.datetime.combine(date, time1)
    datetime2 = datetime.datetime.combine(date, time2)
    return chop_microseconds(datetime1 - datetime2)


def datetime_median(datetime_arr):
    """
    Find the median of a list of datetime objects.
    :param datetime_arr: numpy array containing datetime objects
    :return: the median datetime object
    """
    if datetime_arr.shape[0] == 0:
        raise ValueError("Array is empty.")
    dates = np.sort(datetime_arr)
    middle = len(dates)//2
    if len(dates) % 2 == 1:
        return dates[middle]
    else:
        return dates[middle-1] + (dates[middle] - dates[middle-1])/2


def calculate_subcost(r, s, time_threshold=datetime.timedelta(seconds=3600), distance_threshold=5):
    """
    Helper function of edit_distance_real (EDR). Calculate subcost between two trajectory points, such that if the two
    points happened within a time threshold, and is away from each other within a distance threshold, then the subcost
    returned is 0. Otherwise, the subcost is assigned to be 1.
    :param r: tuple of (time, lat, long), corresponding to a point in a trajectory. Coordinates in epsg 3857.
    :param s: tuple of (time, lat, long), corresponding to a point in a trajectory. Coordinates in epsg 3857.
    :param time_threshold: timedelta object defining the time threshold.
    :param distance_threshold: distance threshold in metres. Distance is the geodesic distance between points on earth.
    :return: subcost between r and s
    """
    if r[0] > s[0]:
        time_diff = time_difference(r[0], s[0])
    else:
        time_diff = time_difference(s[0], r[0])

    if time_diff <= time_threshold and geodesic_distance(r[1:], s[1:], is_3857=True) <= distance_threshold:
        return 0
    else:
        return 1


def edit_distance_real(R, S, time_threshold=datetime.timedelta(seconds=3600), distance_threshold=5):
    """
    Compute the edit distance on real sequences.
    :param R: List of tuples in the form of (time, lat, long). First real sequence.
    :param S: List of tuples in the form of (time, lat, long). Second real sequence.
    :param time_threshold: timedelta object defining the time threshold.
    :param distance_threshold: distance threshold in metres. Distance is the geodesic distance between points on earth.
    :return: edit distance on real sequences between R and S.
    """
    m = len(R)
    n = len(S)

    memo = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                memo[i][j] = j
            elif j == 0:
                memo[i][j] = i
            else:
                memo[i][j] = min(memo[i-1][j-1] + calculate_subcost(R[i-1], S[j-1], time_threshold, distance_threshold),
                                 memo[i-1][j] + 1,
                                 memo[i][j-1] + 1)

    return memo[m][n]
