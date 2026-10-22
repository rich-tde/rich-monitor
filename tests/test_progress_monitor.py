"""Offline checks: PYTHONPATH=scripts python -m unittest discover -s tests."""

import csv
import json
import shlex
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import progress_monitor as monitor


class ProgressMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.account = patch.object(monitor, "SNELLIUS_ACCOUNT", "test-project")
        self.account.start()
        self.addCleanup(self.account.stop)

    def test_query_includes_all_project_users(self):
        records = [
            ("1", "alice", "COMPLETED", "2026-03-01T01:00:00"),
            ("2", "bob", "RUNNING", "2026-03-01T02:00:00"),
            ("2.batch", "bob", "RUNNING", "2026-03-01T02:00:00"),
            ("3", "bob", "PENDING", "Unknown"),
        ]
        lines = []
        for job_id, user, state, start in records:
            record = {
                "JobID": job_id,
                "User": user,
                "JobName": "same-run",
                "State": state,
                "Start": start,
                "End": "Unknown",
                "Elapsed": "01:00:00",
                "AllocCPUS": "4",
                "NNodes": "1",
            }
            lines.append("|".join(record[field] for field in monitor.SACCT_FIELDS))
        response = subprocess.CompletedProcess([], 0, "\n".join(lines), "")
        with patch.object(monitor.subprocess, "run", return_value=response) as run:
            jobs = monitor._fetch_sacct_chunk(date(2026, 3, 1), date(2026, 3, 7))
        command = shlex.split(run.call_args.args[0][-1])
        self.assertIn("--allusers", command)
        self.assertIn("--accounts=test-project", command)
        self.assertNotIn("-u", command)
        self.assertEqual([job["User"] for job in jobs], ["alice", "bob"])
        self.assertEqual([job["CoreHours"] for job in jobs], [4.0, 4.0])

    def test_failed_query_is_distinct_from_empty_results(self):
        with patch.object(
            monitor.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 1, "", "SSH failed"),
        ):
            self.assertIsNone(
                monitor._fetch_sacct_chunk(date(2026, 3, 1), date(2026, 3, 7))
            )

    def test_cache_migration_reuse_and_account_change(self):
        path = self.output / monitor.SACCT_CACHE_FNAME
        path.write_text(json.dumps({"2026-03-01_2026-03-07": [{"User": "old"}]}))
        past = (date(2026, 3, 1), date(2026, 3, 7))
        current = (date(2026, 3, 8), date(2026, 3, 8))
        with (
            patch.object(monitor, "date") as clock,
            patch.object(monitor, "_chunk_dates", return_value=[past, current]),
            patch.object(monitor, "_fetch_sacct_chunk") as fetch,
        ):
            clock.today.return_value = date(2026, 3, 8)
            fetch.side_effect = [
                [{"JobID": "1", "Start": "2026-03-01", "User": "alice"}],
                [{"JobID": "2", "Start": "2026-03-01", "User": "bob"}],
            ]
            jobs = monitor._get_all_jobs(str(self.output))
            self.assertEqual(
                jobs,
                [
                    {"JobID": "1", "Start": "2026-03-01", "User": "alice"},
                    {"JobID": "2", "Start": "2026-03-01", "User": "bob"},
                ],
            )
            self.assertEqual(fetch.call_count, 2)
            stored = json.loads(path.read_text())
            self.assertEqual(stored["scope"]["account"], "test-project")
            self.assertEqual(len(stored["chunks"]), 1)

            fetch.reset_mock()
            fetch.side_effect = [
                [{"JobID": "3", "Start": "2026-03-01", "User": "carol"}]
            ]
            jobs = monitor._get_all_jobs(str(self.output))
            fetch.assert_called_once_with(*current)
            self.assertEqual(
                jobs,
                [
                    {"JobID": "1", "Start": "2026-03-01", "User": "alice"},
                    {"JobID": "3", "Start": "2026-03-01", "User": "carol"},
                ],
            )

            fetch.reset_mock()
            fetch.side_effect = [[], []]
            with patch.object(monitor, "SNELLIUS_ACCOUNT", "another-project"):
                self.assertEqual(monitor._get_all_jobs(str(self.output)), [])
            self.assertEqual(fetch.call_count, 2)

    def test_failed_window_is_retried(self):
        with (
            patch.object(monitor, "date") as clock,
            patch.object(
                monitor,
                "_chunk_dates",
                return_value=[(date(2026, 3, 1), date(2026, 3, 7))],
            ),
            patch.object(
                monitor, "_fetch_sacct_chunk", side_effect=[None, []]
            ) as fetch,
        ):
            clock.today.return_value = date(2026, 3, 8)
            self.assertEqual(monitor._get_all_jobs(str(self.output)), [])
            self.assertFalse((self.output / monitor.SACCT_CACHE_FNAME).exists())
            self.assertEqual(monitor._get_all_jobs(str(self.output)), [])
            self.assertEqual(fetch.call_count, 2)
            self.assertTrue((self.output / monitor.SACCT_CACHE_FNAME).exists())

    def test_job_spanning_windows_is_counted_once_with_latest_state(self):
        old = {
            "JobID": "42",
            "Start": "2026-03-07",
            "User": "alice",
            "State": "RUNNING",
            "CoreHours": 4.0,
        }
        latest = {**old, "State": "COMPLETED", "CoreHours": 8.0}
        reused_id = {**latest, "Start": "2026-03-08"}
        with (
            patch.object(monitor, "date") as clock,
            patch.object(
                monitor,
                "_chunk_dates",
                return_value=[
                    (date(2026, 3, 1), date(2026, 3, 7)),
                    (date(2026, 3, 8), date(2026, 3, 8)),
                ],
            ),
            patch.object(
                monitor,
                "_fetch_sacct_chunk",
                side_effect=[
                    [old],
                    [latest, reused_id],
                    [latest, reused_id],
                ],
            ),
        ):
            clock.today.return_value = date(2026, 3, 8)
            self.assertEqual(
                monitor._get_all_jobs(str(self.output)), [latest, reused_id]
            )
            # A cached earlier window must not reintroduce the old record.
            self.assertEqual(
                monitor._get_all_jobs(str(self.output)), [latest, reused_id]
            )

    def test_report_separates_users_and_keeps_project_total(self):
        jobs = [
            {
                "JobID": "1",
                "User": "alice",
                "JobName": "same-run",
                "State": "COMPLETED",
                "Start": "2026-03-01",
                "CoreHours": 4.0,
            },
            {
                "JobID": "2",
                "User": "bob",
                "JobName": "same-run",
                "State": "RUNNING",
                "Start": "2026-03-02",
                "CoreHours": 8.0,
            },
            {
                "JobID": "3",
                "User": "alice",
                "JobName": "same-run",
                "State": "COMPLETED",
                "Start": "2026-03-03",
                "CoreHours": 2.0,
            },
        ]
        path = self.output / "progress.csv"
        monitor._write_report(jobs, {}, str(path))
        lines = path.read_text().splitlines()
        rows = list(
            csv.DictReader(line for line in lines if line and not line.startswith("#"))
        )
        self.assertEqual(
            [
                (row["User"], row["Jobs"], row["TotalCoreHours"], row["Running"])
                for row in rows[:-1]
            ],
            [("bob", "1", "8.0", "yes"), ("alice", "2", "6.0", "no")],
        )
        self.assertEqual(rows[-1]["RunName"], "TOTAL")
        self.assertEqual(rows[-1]["User"], "")
        self.assertEqual(rows[-1]["TotalCoreHours"], "14.0")
        self.assertTrue(all(row["SnapNum"] == "" for row in rows))
        self.assertIn("# JobID,User,JobName", path.read_text())


if __name__ == "__main__":
    unittest.main()
