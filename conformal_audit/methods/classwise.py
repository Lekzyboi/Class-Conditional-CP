"""Classwise conformal prediction scaffold."""

from conformal_audit.methods.mondrian import MondrianCP


class ClasswiseCP(MondrianCP):
    name = "classwise_cp"

