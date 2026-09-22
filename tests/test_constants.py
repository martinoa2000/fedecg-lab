"""Tests pinning the dataset constants that results depend on."""

from __future__ import annotations

from fedecg.data import constants


class TestSuperclasses:
    def test_label_order_is_frozen(self):
        """Changing this order silently invalidates every saved metric table."""
        assert constants.SUPERCLASSES == ("NORM", "MI", "STTC", "CD", "HYP")

    def test_num_classes_matches(self):
        assert constants.NUM_CLASSES == len(constants.SUPERCLASSES)

    def test_labels_are_unique(self):
        assert len(set(constants.SUPERCLASSES)) == len(constants.SUPERCLASSES)


class TestSignalShape:
    def test_twelve_leads(self):
        assert constants.NUM_LEADS == 12

    def test_signal_length_is_ten_seconds(self):
        assert constants.SIGNAL_LENGTH == 10 * constants.SAMPLING_RATE_HZ


class TestFolds:
    def test_official_split_covers_all_ten_folds(self):
        folds = {*constants.TRAIN_FOLDS, constants.VAL_FOLD, constants.TEST_FOLD}
        assert folds == set(range(1, 11))

    def test_splits_are_disjoint(self):
        assert constants.VAL_FOLD not in constants.TRAIN_FOLDS
        assert constants.TEST_FOLD not in constants.TRAIN_FOLDS
        assert constants.VAL_FOLD != constants.TEST_FOLD
