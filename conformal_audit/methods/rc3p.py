"""Efficient rare-class grouped conformal prediction.

This is a documented practical substitute for RC3P until a faithful RC3P
implementation is added. It groups classes by calibration or training
frequency and reuses the ClusteredCP implementation.
"""

from conformal_audit.methods.clustered import ClusteredCP


class RC3P(ClusteredCP):
    name = "rc3p"

    def diagnostics(self) -> dict[str, object]:
        diagnostics = super().diagnostics()
        diagnostics["method"] = self.name
        diagnostics["implementation_note"] = "frequency-clustered substitute for RC3P"
        return diagnostics
