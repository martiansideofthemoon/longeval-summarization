import csv
from datetime import datetime
import numpy as np
from typing import Dict
from itertools import tee


MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "July": 7,
    "Jul": 7,
    "Aug": 8,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    See https://docs.python.org/3.8/library/itertools.html#itertools-recipes
    """
    x, y = tee(iterable)
    next(y, None)
    return zip(x, y)

def parse_datetime(datetime_str):
    """ Parse a datetime string in MTurk format """
    _, month, day, time, _, year = datetime_str.split()
    hours, minutes, seconds = time.split(":")

    return datetime(
        int(year), MONTHS[month], int(day), int(hours), int(minutes), int(seconds),
    )

def extract_worker_stats(mturk_data, time_threshold=0):
    """ Extract worker stats """

    def meets_threshold(median_time):
        """ Simple time threshold based on median time taken on HIT """
        return not time_threshold or median_time > time_threshold

    unique_workers: Dict[str, int] = {}
    worker_accept_times = {}
    worker_submit_times = {}
    worker_calc_times = {}
    for row in mturk_data:
        worker_id = row["WorkerId"]
        unique_workers[worker_id] = unique_workers.get(worker_id, 0) + 1

        accept_times = worker_accept_times.get(worker_id, [])
        accept_times.append(parse_datetime(row["AcceptTime"]))
        worker_accept_times[worker_id] = accept_times

        submit_times = worker_submit_times.get(worker_id, [])
        submit_times.append(parse_datetime(row["SubmitTime"]))
        worker_submit_times[worker_id] = submit_times

        worktimesecs = worker_calc_times.get(worker_id, [])
        worktimesecs.append(int(row["WorkTimeInSeconds"]))
        worker_calc_times[worker_id] = worktimesecs

    time_stats: Dict[str, Dict[str, float]] = {}
    for worker_id, submit_times in worker_submit_times.items():
        submit_times = sorted(submit_times)
        accept_times = sorted(worker_accept_times[worker_id])
        total_time_diff = (submit_times[-1] - accept_times[0]).total_seconds()
        if len(submit_times) > 1:
            time_diffs = [
                (end - start).total_seconds()
                for start, end in pairwise(sorted(submit_times))
            ]
            median = np.median(time_diffs)
            if meets_threshold(median):
                time_stats[worker_id] = {
                    "std": np.std(time_diffs),
                    "mean": np.mean(time_diffs),
                    "median": median,
                    "median_first_five": np.median(time_diffs[:5]),
                    "time_diffs": time_diffs,
                    "submits": len(submit_times),
                    "total_time": total_time_diff,
                    "WorkTimeSeconds": worker_calc_times[worker_id]
                }
        elif meets_threshold(total_time_diff):
            print("yolo")
            time_stats[worker_id] = {
                "std": 0,
                "mean": total_time_diff,
                "median": total_time_diff,
                "submits": 1,
                "time_diffs": [total_time_diff],
                "total_time": total_time_diff,
                "WorkTimeSeconds": worker_calc_times[worker_id]
            }

    histogram = {}
    for bucket in pairwise((1, 2, 10, 25, 50, float("inf"))):
        start = str(bucket[0])
        end = str(bucket[1])
        key = f"[{start}]" if start == end else f"[{start}-{end})"
        histogram[key] = sum(
            1 for c in unique_workers.values() if c >= bucket[0] and c < bucket[1]
        )

    return histogram, time_stats, unique_workers.keys()