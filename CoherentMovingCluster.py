from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import math


class CoherentMovingCluster:

    TIME_INTERVAL = 5

    def __init__(self, min_obj, min_lifetime, dist_threshold, metric_func):
        self.min_obj = min_obj
        self.min_lifetime = min_lifetime
        self.dist_threshold = dist_threshold
        self.CLUSTERING_DB = DBSCAN(self.dist_threshold, self.min_obj, metric=metric_func)

    def offline_cmc(self, traj_df):
        """
        An offline algorithm to detect convoys from raw locations data stored in a DataFrame.
        :param traj_df: a pandas DataFrame with columns ['TIMESTAMP', 'USERID', 'LONGITUDE', 'LATITUDE']
        :return: a list of convoy objects detected
        """
        traj_df_columns = list(traj_df)
        if traj_df_columns[0] != 'TIMESTAMP' or traj_df_columns[1] != 'USERID' or \
                traj_df_columns[2] != 'LONGITUDE' or traj_df_columns[3] != 'LATITUDE':
            raise ValueError("Input is not a pandas DataFrame with columns ['TIMESTAMP', 'USERID', 'LONGITUDE', \
            'LATITUDE']")

        vs = []
        v_result = []

        current_time = traj_df['TIMESTAMP'].min()
        end_time = traj_df['TIMESTAMP'].max()
        o_last = pd.DataFrame(columns=['TIMESTAMP', 'USERID', 'LONGITUDE', 'LATITUDE'])

        while True:
            # Obtain the data points for the current time interval
            v_next = []
            o_next = traj_df[(traj_df['TIMESTAMP'] >= current_time) &
                             (traj_df['TIMESTAMP'] < (current_time + CoherentMovingCluster.TIME_INTERVAL))]
            o_next = o_next.loc[o_next.groupby('USERID').TIMESTAMP.idxmax()]    # retain only latest row per user
            temp_o = o_next
            lost_users = list(set(o_last['USERID'].unique()) - set(o_next['USERID'].unique()))
            if len(lost_users):
                o_next = pd.concat([o_next, o_last[o_last['USERID'].isin(lost_users)]])

            # detect convoys for the current time interval
            if len(o_next.index) < self.min_obj:
                convoys = []
            else:
                convoys = self.dbscan_to_convoys(o_next, traj_df)

            for v in vs:
                v.assigned = False

                for c in convoys:
                    # check if new convoys intersect with old convoys.
                    if len(np.intersect1d(c.oids[0], v.oids[-1])) >= self.min_obj:
                        v.assigned = True
                        v.oids += c.oids
                        v.rids += c.rids
                        v.end_time = current_time + CoherentMovingCluster.TIME_INTERVAL - 1
                        v_next.append(v)
                        c.assigned = True

                if not v.assigned and v.get_lifetime() >= self.min_lifetime:
                    # convoy ends. Social distancing enforced.
                    v_result.append(v)

            for c in convoys:
                # new convoys detected
                if not c.assigned:
                    c.start_time = current_time
                    c.end_time = current_time + CoherentMovingCluster.TIME_INTERVAL - 1
                    v_next.append(c)

            vs = v_next
            o_last = temp_o
            current_time += CoherentMovingCluster.TIME_INTERVAL     # go to next time interval

            if current_time > end_time:
                # all data points in dataframe processed
                # check all un-ended convoys in vs and store to result of lifetime is more than the minimum
                for v in vs:
                    if v.get_lifetime() >= self.min_lifetime:
                        v_result.append(v)

                return v_result

    def dbscan_to_convoys(self, object_df, all_df):
        """
        Use DBScan detect convoys and create Convoy objects if detected
        :param object_df: a Pandas DataFrame containing the locations for the current time interval
        :param all_df: a Pandas DataFrame containing all locations in the data set
        :return: a list of Convoy objects detected
        """
        convoys = []

        # fit using DBScan
        labels = self.CLUSTERING_DB.fit_predict(object_df)

        for label in np.unique(labels):
            if label != -1:
                # get data points of each detected convoy and create new Convoy object.
                c_objects = np.where(labels == label)
                new_convoy = Convoy()
                new_convoy.oids = [np.array(object_df.iloc[c_objects[0]]['USERID'])]
                new_convoy.rids = [[all_df.index.get_loc(obj.name) for _, obj in object_df.iloc[c_objects[0]].iterrows()]]

                convoys.append(new_convoy)

        return convoys


class Convoy:

    def __init__(self):
        self.assigned = False
        self.oids = []  # (moving) object ids at each time interval
        self.rids = []  # row ids in data frame at each time interval
        self.start_time = -1
        self.end_time = -1

    def get_lifetime(self):
        """
        Calculate the duration of the convoy lifetime
        :return: Duration of the convoy lifetime in seconds
        """
        return self.end_time - self.start_time

    def to_dict(self):
        """
        Convert Convoy object in dictionary form
        :return: dictionary object representing the Convoy object
        """
        return {
            'oids': self.oids,
            'rids': self.rids,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
