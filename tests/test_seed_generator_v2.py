"""
SeedGenerator v2 (nested Repeat) unit-test
-----------------------------------------
AST 구조:

root
 ├─ arm   (hash:A) ── screw (hash:B)
 ├─ arm   (hash:A) ── screw (hash:B)
 └─ arm   (hash:A) ── screw (hash:B)

Expect :
  • 1st-level Repeat(count=3, template=arm subtree)
  • inside template another Repeat(count=3, template=screw)
"""
from gpe_core.ast_builder import ASTBuilder
from gpe_core.repetition_detector import RepetitionDetector
from gpe_core.seed_generator import SeedGenerator
from gpe_core.models import RepeatRule

def _mock_obj():
    screw = {"type": "screw", "len": 5}
    arm   = {"parts": [screw, screw, screw]}
    return {"robot": [arm, arm, arm]}

def test_nested_repeat_structure():
    builder = ASTBuilder(); root = builder.build(_mock_obj())
    reps = RepetitionDetector(builder.nodes, min_occ=2, min_size=2).detect()
    seeds = SeedGenerator(builder.nodes, reps).generate(root)
    rules = seeds[0].rules

    # top-level must be single Repeat(count=3)
    assert isinstance(rules[0], RepeatRule)
    assert rules[0].count == 3

    inner = rules[0].instruction
    # inside template must have NEW(arm) then Repeat(count=3) for screws
    assert any(isinstance(r, RepeatRule) and r.count == 3 for r in inner)
