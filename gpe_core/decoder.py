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
    """Raised when a GPE payload cannot be decoded."""
    pass

# ---------------------------------------------------------------------------
try:
    from numba import njit, typed, types  # type: ignore
    # 환경변수로 제어 가능하게 변경
    import os
    _NUMBA_AVAIL = os.environ.get('GPE_USE_NUMBA', 'true').lower() == 'true'
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA_AVAIL = False

    # ------------------------------------------------------------------
class GPEDecoder:
    def decode(self, payload: GpePayload) -> Any:
        try:
            # fallback JSON with robust error handling
            fb = payload.fallback_payload
            if fb and fb.get("json"):
                try:
                    return json.loads(fb["json"])
                except json.JSONDecodeError as e:
                    raise GPEDecodeError(f"Fallback JSON parsing failed: {e}") from e
                except Exception as e:
                    raise GPEDecodeError(f"Unexpected error in fallback: {e}") from e

            # Validate payload structure
            if not payload.generative_payload:
                raise GPEDecodeError("No generative payload found")
            if "seeds" not in payload.generative_payload:
                raise GPEDecodeError("No seeds in generative payload")
            if "root_id" not in payload.generative_payload:
                raise GPEDecodeError("No root_id in generative payload")

            # prepare containers
            objs: Dict[str, Any] = {}
            meta: Dict[str, Dict[str, Any]] = {}

            # choose apply loop implementation
            if _NUMBA_AVAIL:
                self._apply_numba(payload.generative_payload["seeds"], objs, meta)
            else:
                for seed in payload.generative_payload["seeds"]:
                    for rule in seed.get("rules", []):
                        self._apply_py(rule, objs, meta)

            root_id = payload.generative_payload["root_id"]
            if root_id not in objs:
                raise GPEDecodeError(f"Root object {root_id} not found")
                
            return objs[root_id]
            
        except GPEDecodeError:
            raise
        except Exception as e:
            raise GPEDecodeError(f"Decoding failed: {e}") from e

# ==================================================================
# Pure‑python implementation (fallback)
# ==================================================================
def _apply_py(self, r: Dict[str, Any], o: Dict[str, Any], m: Dict[str, Dict[str, Any]]):
    op = r["op_code"]
    if op == "NEW":
        vid, cls = r["instance_id"], r["class_name"]
        attrs = r.get("attributes", {})
        if "value" in attrs:
            o[vid] = attrs["value"]
        elif cls == "dict":
            o[vid] = {}
        elif cls == "list":
            o[vid] = []
        else:
            # 클래스 정보 보존
            o[vid] = {"__class__": cls, "__type__": "custom"}
        m[vid] = attrs
    elif op == "APPEND":
        p, c = r["parent_id"], r["child_id"]
        if p not in o:
            raise ValueError(f"Parent {p} not found")
        if c not in o:
            raise ValueError(f"Child {c} not found")
            
        parent, child = o[p], o[c]
        if isinstance(parent, list):
            parent.append(child)
        elif isinstance(parent, dict):
            key = m[c].get("key")
            if key is None:
                # 키가 없으면 경고 또는 자동 생성
                import warnings
                warnings.warn(f"No key found for child {c}, skipping append")
            else:
                parent[key] = child
        else:
            raise TypeError(f"Cannot append to {type(parent)}")
    elif op == "REPEAT":
        count = r.get("count", 0)
        if count <= 0:
            return
            
        instruction = r.get("instruction", [])
        for _ in range(count):
            tmpl = copy.deepcopy(instruction)
            if isinstance(tmpl, list):
                for rule in tmpl:
                    self._apply_py(rule, o, m)
            else:
                self._apply_py(tmpl, o, m)
    else:
        raise ValueError(f"Unknown operation: {op}")

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
                                objs[vid] = {"__class__": cls, "__type__": "custom"}
                            meta[vid] = attrs
                        elif op == "APPEND":
                            p = rule["parent_id"]; c = rule["child_id"]
                            parent = objs[p]; child = objs[c]
                            if isinstance(parent, list):
                                parent.append(child)
                            else:
                                key = meta[c].get("key")
                                if key is not None:  # key 체크 추가
                                    parent[key] = child
                        elif op == "REPEAT":
                            for _ in range(rule["count"]):
                                    instruction = [instruction]
                                for sub in instruction:
                                    # NOTE: recursion depth typically shallow; inline for perf
                                    sop = sub["op_code"]
                                    if sop == "NEW":
                                        vid = sub["instance_id"]
                                        cls = sub["class_name"]
                                        attrs = sub.get("attributes", {})
                                        # Python 경로와 일관성 있게 수정
                                        if "value" in attrs:
                                            objs[vid] = attrs["value"]
                                        elif cls == "dict":
                                            objs[vid] = {}
                                        elif cls == "list":
                                            objs[vid] = []
                                        else:
                                            objs[vid] = {"__class__": cls, "__type__": "custom"}
                                        meta[vid] = attrs
                                    elif sop == "APPEND":
                                        pp = sub["parent_id"]
                                        cc = sub["child_id"]
                                        pobj = objs[pp]
                                        if isinstance(pobj, list):
                                            pobj.append(objs[cc])
                                        else:
                                            # meta에서 key 가져오기
                                            key = meta[cc].get("key")
                                            if key is not None:
                                              pobj[key] = objs[cc]
            run(seeds, t_objs, t_meta)
            # move back to python dict for downstream
            objs_py.update(t_objs)
            meta_py.update(t_meta)

    else:
        def _apply_numba(self, *a, **kw):  # type: ignore[no-redef]
            raise RuntimeError("Numba not installed")
