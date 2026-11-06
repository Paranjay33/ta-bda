"""The ONE text pipeline used for live and batch inference.

raw user text
  -> shared normalization contract   (hinglish.normalization; same as training data)
  -> dataset-style transform         (hinglish.dataset_style; maps raw text into the
                                       lowercased/stemmed/stopword-filtered space the
                                       supplied training data lives in)
  -> model input (identical for E1, E2, E3, E4)
"""
from __future__ import annotations

from dataclasses import dataclass

from hinglish.config import ProjectConfig
from hinglish.dataset_style import DatasetStyleConfig, to_dataset_style
from hinglish.normalization import NormalizationConfig, normalize_text


@dataclass(frozen=True)
class PreparedText:
    raw: str
    normalized: str      # shared normalization contract output
    model_input: str     # what every model actually receives


class TextPipeline:
    def __init__(self, cfg: ProjectConfig):
        self.norm_cfg = NormalizationConfig.from_project_config(cfg.get("preprocessing", default={}))
        self.style_cfg = DatasetStyleConfig.from_project_config(cfg)
        self.version = f"{cfg.get('preprocessing', 'version', default='shared-norm')}+{self.style_cfg.version if self.style_cfg.enabled else 'no-style'}"

    def prepare(self, text: str) -> PreparedText:
        normalized = normalize_text(text, self.norm_cfg)
        return PreparedText(raw=text, normalized=normalized, model_input=to_dataset_style(normalized, self.style_cfg))
