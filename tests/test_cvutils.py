#!/usr/bin/env python

"""Tests for `cvutils` package."""

from cvutils.cvutils import sample


def test_sample():
    assert sample(True)
    assert not sample(False)
