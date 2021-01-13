from CoherentMovingCluster import *
from BisectingKMeans import *
from utils import *
import utils


def test_cmc():
    ar = np.array([[0, 0, 0, 0], [1, 0, 1, 1], [2, 0, 0, 2], [3, 0, 0, 1], [4, 0, 1, 0], [5, 0, 1, 1], [6, 0, 0, 2],
                   [0, 1, 0, 0], [1, 1, 0, 1], [2, 1, 1, 2], [3, 1, 0, 3], [4, 1, 1, 2], [5, 1, 2, 2], [6, 1, 1, 3],
                   [0, 2, 0, 0], [1, 2, 1, 1], [2, 2, 2, 2], [3, 2, 3, 3], [4, 2, 3, 4], [5, 2, 3, 3], [6, 2, 4, 3]])
    df = pd.DataFrame(ar, columns=['TIMESTAMP', 'USERID', 'LONGITUDE', 'LATITUDE'])

    def row_euclidean(row1, row2):
        return math.sqrt((row1[2] - row2[2])**2 + (row1[3] - row2[3])**2)

    CoherentMovingCluster.TIME_INTERVAL = 1
    cmc = CoherentMovingCluster(2, 2, 2, row_euclidean)
    convoys_list = cmc.offline_cmc(df)

    assert len(convoys_list) == 1
    assert len(convoys_list[0].oids) == 7
    assert len(convoys_list[0].rids) == 7
    assert convoys_list[0].start_time == 0
    assert convoys_list[0].end_time == 6


def test_wcss():
    bi_kmeans = BisectingKMeans()
    bi_kmeans.X = pd.DataFrame(np.array([[3, 1], [1, 0], [2, 1], [0, 1], [1, 2], [0, 1]]))
    bi_kmeans.labels_ = np.array([0, 0, 1, 1, 2, 2])
    bi_kmeans.current_n_clusters = 3
    results = bi_kmeans.wcss()

    assert results.shape[0] == 3
    assert results[0] == 2.5
    assert results[1] == 2.0
    assert results[2] == 1.0


def test_geodesic_distance():
    kuala_lumpur_4326 = (3.138945, 101.687140)
    queensland_4326 = (-19.442673, 145.803937)

    kuala_lumpur_3857 = (349600.677884, 11319760.691222)
    queensland_3857 = (-2207123.495036, 16230820.011363)

    temp_kl = utils.TRANSFORMER.transform(kuala_lumpur_3857[1], kuala_lumpur_3857[0])
    assert abs(temp_kl[0] - kuala_lumpur_4326[0]) < 1e-4
    assert abs(temp_kl[1] - kuala_lumpur_4326[1]) < 1e-4

    temp_ql = utils.TRANSFORMER.transform(queensland_3857[1], queensland_3857[0])
    assert abs(temp_ql[0] - queensland_4326[0]) < 1e-4
    assert abs(temp_ql[1] - queensland_4326[1]) < 1e-4

    assert abs(geodesic_distance(kuala_lumpur_4326, queensland_4326) -
               geodesic_distance(kuala_lumpur_3857, queensland_3857, is_3857=True)) < 0.1


def test_chop_microseconds():
    input_delta = datetime.timedelta(seconds=1, microseconds=672000)
    expected = datetime.timedelta(seconds=1)
    assert chop_microseconds(input_delta) == expected


def test_time_difference():
    input1 = datetime.time(13, 34, 35, 1234)
    input2 = datetime.time(0, 19, 58, 4321)
    expected = datetime.timedelta(hours=13, minutes=14, seconds=36)

    assert time_difference(input1, input2) == expected


def test_datetime_median():
    datetime_arr = np.array([
        datetime.datetime(year=2020, month=10, day=8, hour=13),
        datetime.datetime(year=2020, month=10, day=8, hour=22),
        datetime.datetime(year=2020, month=10, day=8, hour=9),
        datetime.datetime(year=2020, month=10, day=8, hour=7),
        datetime.datetime(year=2020, month=10, day=8, hour=12),
        datetime.datetime(year=2020, month=10, day=8, hour=21)
    ])
    assert datetime_median(datetime_arr) == datetime.datetime(year=2020, month=10, day=8, hour=12, minute=30)

    datetime_arr = np.array([
        datetime.datetime(year=2020, month=10, day=8, hour=13),
        datetime.datetime(year=2020, month=10, day=8, hour=22),
        datetime.datetime(year=2020, month=10, day=8, hour=9),
        datetime.datetime(year=2020, month=10, day=8, hour=7),
        datetime.datetime(year=2020, month=10, day=8, hour=12),
    ])
    assert datetime_median(datetime_arr) == datetime.datetime(year=2020, month=10, day=8, hour=12)


def test_calculate_subcost():
    kl = (datetime.time(hour=22), 349600, 11319760)
    kl2 = (datetime.time(hour=22), 349601, 11319760)
    kl3 = (datetime.time(hour=20), 349601, 11319760)
    ql = (datetime.time(hour=22), -2207123, 16230820)

    assert calculate_subcost(kl, kl2, distance_threshold=1000) == 0
    assert calculate_subcost(kl, kl3, distance_threshold=1000) == 1
    assert calculate_subcost(kl, ql, distance_threshold=1000) == 1


def test_edit_distance_real():
    traj1 = [
        (datetime.time(hour=22), 349600, 11319760),
        (datetime.time(hour=22, second=38), 349597, 11319760),
        (datetime.time(hour=22, second=48), 349597, 11319753),
        (datetime.time(hour=22, minute=1, second=5), 349591, 11319753),
        (datetime.time(hour=22, minute=1, second=44), 349587, 11319757)
    ]
    traj2 = [
        (datetime.time(hour=22), 349600, 11319760),
        (datetime.time(hour=22, second=8), 349594, 11319754),
        (datetime.time(hour=22, second=30), 349588, 11319754),
        (datetime.time(hour=22, second=39), 349583, 11319759),
        (datetime.time(hour=22, minute=1, second=4), 349588, 11319759),
        (datetime.time(hour=22, minute=1, second=49), 349582, 11319759),
        (datetime.time(hour=22, minute=1, second=59), 349583, 11319759)
    ]

    for x in traj1:
        for y in traj2:
            print(calculate_subcost(x, y, time_threshold=datetime.timedelta(seconds=30), distance_threshold=10))

    # add traj2[4];  add traj2[6];
    assert edit_distance_real(traj1, traj2, time_threshold=datetime.timedelta(seconds=30), distance_threshold=10) == 2

    # traj1[1] -> traj2[1]; add traj2[3]; add traj2[6];
    assert edit_distance_real(traj1, traj2, time_threshold=datetime.timedelta(seconds=20), distance_threshold=10) == 3


if __name__ == "__main__":
    test_cmc()
    test_wcss()
    test_geodesic_distance()
    test_chop_microseconds()
    test_time_difference()
    test_datetime_median()
    test_calculate_subcost()
    test_edit_distance_real()
