"""NTU feeder for the neuro-symbolic model.

The class-concept prior is owned by the symbolic compatibility module. The
feeder therefore returns the same four-tuple as the official NTU feeder and
does not attach a duplicated per-sample copy of the class prior.
"""

from __future__ import annotations

from feeders.feeder_ntu import Feeder as BaseFeeder


class Feeder(BaseFeeder):
    def __init__(self, concept_csv: str | None = None, eval_mode: bool = False, **kwargs):
        self.eval_mode = eval_mode
        super().__init__(**kwargs)
        # Kept as an accepted argument for old configs, but intentionally not
        # loaded. The symbolic module is the single source of truth for the
        # class-concept support matrix.
        self.concept_csv = concept_csv

    def __getitem__(self, index):
        if self.eval_mode and self.split == "train":
            original_split = self.split
            self.split = "train_eval"
            try:
                data, index_t, label, sample_index = super().__getitem__(index)
            finally:
                self.split = original_split
        else:
            data, index_t, label, sample_index = super().__getitem__(index)
        return data, index_t, label, sample_index
