"""Tests for training progress and ETA reporting."""

from utils.log import TrainingProgress, format_duration


def test_training_progress_percentage_and_eta():
    progress = TrainingProgress(total_steps=100, start_step=20, start_time=10.0)
    stats = progress.snapshot(completed_steps=40, now=20.0)
    assert stats["percent"] == 40.0
    assert stats["steps_per_second"] == 2.0
    assert stats["eta_seconds"] == 30.0


def test_format_duration():
    assert format_duration(65) == "01m 05s"
    assert format_duration(90061) == "1d 01h 01m 01s"
    assert format_duration(float("inf")) == "unknown"
