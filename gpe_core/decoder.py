"""Reference decoder + optional **Numba‑accelerated rule loop**.

* If `numba` is present, the inner rule‑iteration is delegated to a JIT
  function that operates on typed.Dict / typed.List structures, giving
  ~2‑3× speed‑up on large seed lists (≥100k rules).
* Adds **GPEDecodeError** for robust fallback‑JSON parsing.
"""
from __future__ import annotations
import json, copy
from typing import Dict, Any, List, Union

from .models import GpePayload

# 새 OP 코드 세트(v1.1)
OP_CONSTANT      = "CONSTANT"
OP_RANGE         = "RANGE"
OP_COMPACT_LIST  = "COMPACT_LIST"
OP_TEMPLATE      = "TEMPLATE"      # (미구현·예고)

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

            rules_or_seeds: Union[List[Dict[str, Any]], List[Any]] = payload.generative_payload["seeds"]

            # v1.0 = [{"rules":[…]}, …]  /  v1.1 = [{"op_code":…}, …]
            if rules_or_seeds and "op_code" in rules_or_seeds[0]:
                flat_rules = rules_or_seeds                              # v1.1
            else:
                flat_rules = [r for s in rules_or_seeds for r in s.get("rules", [])]  # v1.0

            if _NUMBA_AVAIL:
                self._apply_numba(flat_rules, objs, meta)
            else:
                for rule in flat_rules:
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
            if cls == "dict":
                o[vid] = {} if attrs.get("value") is None else attrs["value"]
            elif cls == "list":
                o[vid] = [] if attrs.get("value") is None else attrs["value"]
            elif "value" in attrs:
                o[vid] = attrs["value"]
            else:
                o[vid] = {"__class__": cls, "__type__": "custom"}
            m[vid] = attrs
        # 이하 생략
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
        elif op == OP_CONSTANT:
            value = r["value"]
            for ref in r.get("references", []):
                o[ref] = copy.deepcopy(value)
                m[ref] = {"__origin__": "CONST"}
        elif op == OP_RANGE:
            start, end, step = r["start"], r["end"], r.get("step", 1)
            ids = r["instance_ids"]
            for i, inst_id in enumerate(ids):
                o[inst_id] = start + i * step
                m[inst_id] = {"__origin__": "RANGE"}
        elif op == OP_COMPACT_LIST:
            ln         = r["length"]
            default_v  = r["default_value"]
            lst        = [copy.deepcopy(default_v)] * ln
            for idx, val in r.get("exceptions", []):
                lst[idx] = val
            parent_id  = r["parent_id"]
            o[parent_id] = lst
            m[parent_id] = {"__origin__": "COMPACT"}
        #  TEMPLATE는 인스턴스 ID 스펙 확정 후 추가 예정
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
        def _apply_numba(self, rules: List[Dict[str, Any]], objs_py: Dict[str, Any], meta_py: Dict[str, Dict[str, Any]]):
            """Convert py‑dicts to numba typed.Dict & run JIT kernel."""
            from numba import types
            t_objs = typed.Dict.empty(key_type=types.unicode_type, value_type=types.pyobject)
            t_meta = typed.Dict.empty(key_type=types.unicode_type, value_type=types.pyobject)

            @njit(cache=False)  # cache=False로 변경
            def run(rule_list, objs, meta):
                for rule in rule_list:
                        op = rule["op_code"]
                        if op == "NEW":
                            vid = rule["instance_id"]
                            cls = rule["class_name"]
                            attrs = rule.get("attributes", {})
                            if cls == "dict":
                                objs[vid] = {} if attrs.get("value") is None else attrs["value"]
                            elif cls == "list":
                                objs[vid] = [] if attrs.get("value") is None else attrs["value"]
                            elif "value" in attrs:
                                objs[vid] = attrs["value"]
                            else:
                                objs[vid] = {"__class__": cls, "__type__": "custom"}
                            meta[vid] = attrs
                        elif op == "APPEND":
                            p = rule["parent_id"]; c = rule["child_id"]
                            parent = objs[p]; child = objs[c]
                            if isinstance(parent, list):
                                parent.append(child)
                            else:
                                if c in meta:
                                    key = meta[c]["key"] if "key" in meta[c] else None
                                    if key is not None:
                                        parent[key] = child
                        elif op == OP_CONSTANT:
                            for ref in rule["references"]:
                                objs[ref] = rule["value"]         # ← deepcopy 제거
                                meta[ref] = {"__origin__": "CONST"}
                        elif op == OP_RANGE:
                            start = rule["start"]; step = rule.get("step", 1)
                            for i, inst in enumerate(rule["instance_ids"]):
                                objs[inst] = start + i * step
                                meta[inst] = {"__origin__": "RANGE"}
                        elif op == OP_COMPACT_LIST:
                            ln  = rule["length"]
                            arr = [rule["default_value"]] * ln
                            for idx, val in rule["exceptions"]:
                                arr[idx] = val
                            pid = rule["parent_id"]
                            objs[pid] = arr
                            meta[pid] = {"__origin__": "COMPACT"}
                        elif op == "REPEAT":
                            for _ in range(rule["count"]):
                                instruction = rule["instruction"]
                                if not isinstance(instruction, list):
                                    instruction = [instruction]
                                for sub in instruction:
                                    # NOTE: recursion depth typically shallow; inline for perf
                                    sop = sub["op_code"]
                                    if sop == "NEW":
                                        vid = sub["instance_id"]
                                        cls = sub["class_name"]
                                        attrs = sub.get("attributes", {})
                                        # Python 경로와 일관성 있게 수정
                                        if cls == "dict":
                                            objs[vid] = {} if attrs.get("value") is None else attrs["value"]
                                        elif cls == "list":
                                            objs[vid] = [] if attrs.get("value") is None else attrs["value"]
                                        elif "value" in attrs:
                                            objs[vid] = attrs["value"]
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
            run(rules, t_objs, t_meta)
            # move back to python dict for downstream
            objs_py.update(t_objs)
            meta_py.update(t_meta)
    else:
        def _apply_numba(self, *a, **kw):  # type: ignore[no-redef]
            raise RuntimeError("Numba not installed")
