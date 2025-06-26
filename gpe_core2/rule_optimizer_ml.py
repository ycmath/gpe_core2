# gpe_core2/rule_optimizer_ml.py
from gpe_core2.rule_optimizer import RuleOptimizerLite
import torch

class RuleOptimizerML(RuleOptimizerLite):
    def __init__(self, pt_path: str):
        super().__init__()
        self.model = torch.jit.load(pt_path).eval()

    def select_rules(self, data, hints=None):
        feats = self._build_features(data, hints)
        logits = self.model(feats).detach().cpu().numpy()
        # rule_rank = np.argsort(-logits) ...
        return self._logits_to_rules(logits)

    # _build_features / _logits_to_rules 는 기존 휴리스틱과 공유가능
