import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from reliability.benchmark import T1, T2, evaluation_data, training_data


FIELDS = ["user_id", "parent_asin", "rating", "timestamp", "history"]


def write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TemporalDataTests(unittest.TestCase):
    def test_unknown_target_stays_in_denominator_and_repeat_is_excluded(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid.csv.gz"
            write_gzip(path, [
                dict(zip(FIELDS, ("u1", "known", 5, T1 + 1, "old"))),
                dict(zip(FIELDS, ("u2", "new", 4, T1 + 2, "old"))),
                dict(zip(FIELDS, ("u3", "known", 5, T1 + 3, "known"))),
                dict(zip(FIELDS, ("u4", "known", 3, T1 + 4, ""))),
            ])
            queries, stats = evaluation_data(path, "valid", {"known", "old"})
            self.assertEqual(len(queries), 2)
            self.assertEqual(stats["new_item_targets"], 1)
            self.assertEqual(stats["seen_target_excluded"], 1)
            self.assertEqual(stats["below_positive_threshold"], 1)

    def test_global_cutoff_rejects_leakage(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv.gz"
            write_gzip(path, [dict(zip(FIELDS, ("u", "item", 5, T1, "")))])
            with self.assertRaisesRegex(ValueError, "t1"):
                training_data(path)
            write_gzip(path, [dict(zip(FIELDS, ("u", "item", 5, T2, "")))])
            with self.assertRaisesRegex(ValueError, "outside"):
                evaluation_data(path, "valid", {"item"})


if __name__ == "__main__":
    unittest.main()
