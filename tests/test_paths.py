"""Tests for project path resolution."""

from __future__ import annotations

from fedecg import paths


class TestProjectRoot:
    def test_points_at_the_repository_root(self):
        assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
        assert (paths.PROJECT_ROOT / "src" / "fedecg").is_dir()

    def test_derived_paths_live_under_the_root(self):
        for path in (
            paths.DATA_DIR,
            paths.RAW_DATA_DIR,
            paths.PTBXL_DIR,
            paths.CONFIG_DIR,
            paths.RESULTS_DIR,
            paths.MLRUNS_DIR,
        ):
            assert path.is_relative_to(paths.PROJECT_ROOT)

    def test_committed_directories_exist(self):
        assert paths.CONFIG_DIR.is_dir()
        assert paths.RESULTS_DIR.is_dir()


class TestEnsureDir:
    def test_creates_nested_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        assert paths.ensure_dir(target) == target
        assert target.is_dir()

    def test_is_idempotent(self, tmp_path):
        target = tmp_path / "already"
        paths.ensure_dir(target)
        paths.ensure_dir(target)
        assert target.is_dir()
