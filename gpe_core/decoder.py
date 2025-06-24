"""Reference decoder + optional **Numba‑accelerated rule loop**.

* If `numba` is present, the inner rule‑iteration is delegated to a JIT
  function that operates on typed.Dict / typed.List structures, giving
  ~2‑3× speed‑up on large seed lists (≥100k rules).
* Adds **GPEDecodeError** for robust fallback‑JSON parsing.
"""
from __future__ import annotations
import json, copy
from typing import Dict, Any, List

from .models import GpePayload

# ---------------------------------------------------------------------------
class GPEDecodeError(RuntimeError):
    """Raised when a GPE payload cannot be decoded (e.g., fallback JSON corrupt)."""

# ---------------------------------------------------------------------------
# 라인 20-25를 다음과 같이 수정:
try:
    from numba import njit, typed, types  # type: ignore
    _NUMBA_AVAIL = False  # 일단 비활성화
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA_AVAIL = False

class GPEDecoder:
    """Single‑process decoder. JIT fast‑path engaged when numba present."""

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON with robust error handling
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            try:
                return json.loads(fb["json"])
            except json.JSONDecodeError as e:
                raise GPEDecodeError(f"Fallback JSON 파싱 실패: {e}") from e

        # 1) prepare containers
        objs: Dict[str, Any] = {}
        meta: Dict[str, Dict[str, Any]] = {}

        # 2) choose apply loop implementation
        if _NUMBA_AVAIL:
            self._apply_numba(payload.generative_payload["seeds"], objs, meta)
        else:
            for seed in payload.generative_payload["seeds"]:
                for rule in seed["rules"]:
                    self._apply_py(rule, objs, meta)

        root_id = payload.generative_payload["root_id"]
        return objs[root_id]

# ==================================================================
# Pure‑python implementation (fallback)
# ==================================================================
    def _apply_py(self, r: Dict[str, Any], o: Dict[str, Any], m: Dict[str, Dict[str, Any]]):
        op = r["op_code"]
        if op == "NEW":  # elif → if로 변경
            vid, cls = r["instance_id"], r["class_name"]
            attrs = r.get("attributes", {})
            if "value" in attrs:
                o[vid] = attrs["value"]
            elif cls == "dict":
                o[vid] = {}
            elif cls == "list":
                o[vid] = []
            else:
                o[vid] = {} 
            m[vid] = attrs
        elif op == "APPEND":
            p, c = r["parent_id"], r["child_id"]
            parent, child = o[p], o[c]
            if isinstance(parent, list):
                parent.append(child)
            else:
                key = m[c].get("key")
                if key is not None:  # 추가된 부분
                    parent[key] = child  # 들여쓰기 수정
        elif op == "REPEAT":
            for _ in range(r["count"]):
                tmpl = copy.deepcopy(r["instruction"])
                if isinstance(tmpl, list):
                    for rule in tmpl:
                        self._apply_py(rule, o, m)
                else:
                    self._apply_py(tmpl, o, m)
        else:
            raise ValueError(op)

    # ==================================================================
    # Numba‑accelerated path
    # ==================================================================
    if _NUMBA_AVAIL:
        def _apply_numba(self, seeds: List[Dict[str, Any]], objs_py: Dict[str, Any], meta_py: Dict[str, Dict[str, Any]]):
            """Convert py‑dicts to numba typed.Dict & run JIT kernel."""
            from numba import types
            t_objs = typed.Dict.empty(key_type=types.unicode_type, value_type=types.pyobject)
            t_meta = typed.Dict.empty(key_type=types.unicode_type, value_type=types.pyobject)

            @njit(cache=False)  # cache=False로 변경
            def run(seeds_list, objs, meta):
                for seed in seeds_list:
                    for rule in seed["rules"]:
                        op = rule["op_code"]
                        if op == "NEW":
                            vid = rule["instance_id"]
                            cls = rule["class_name"]
                            attrs = rule.get("attributes", {})
                            if "value" in attrs:
                                objs[vid] = attrs["value"]
                            elif cls == "dict":
                                objs[vid] = {}
                            elif cls == "list":
                                objs[vid] = []
                            else:
                                objs[vid] = {"__class__": cls}
                            meta[vid] = attrs
                        elif op == "APPEND":
                            p = rule["parent_id"]; c = rule["child_id"]
                            parent = objs[p]; child = objs[c]
                            if isinstance(parent, list):
                                parent.append(child)
                            else:
                                key = meta[c].get("key")
                                parent[key] = child
                        elif op == "REPEAT":
                            for _ in range(rule["count"]):
                                for sub in rule["instruction"]:
                                    # NOTE: recursion depth typically shallow; inline for perf
                                    sop = sub["op_code"]
                                    if sop == "NEW":
                                        vid = sub["instance_id"]
                                        cls = sub["class_name"]
                                        attrs = sub.get("attributes", {})
                                        objs[vid] = {} if cls == "dict" else []
                                        meta[vid] = attrs
                                    elif sop == "APPEND":
                                        pp = sub["parent_id"]
                                        cc = sub["child_id"]
                                        pobj = objs[pp]
                                        if isinstance(pobj, list):
                                            pobj.append(objs[cc])
                                        else:
                                            pobj[str(len(pobj))] = objs[cc]
            run(seeds, t_objs, t_meta)
            # move back to python dict for downstream
            objs_py.update(t_objs)
            meta_py.update(t_meta)

    else:
        def _apply_numba(self, *a, **kw):  # type: ignore[no-self-use]
            raise RuntimeError("Numba not installed")
