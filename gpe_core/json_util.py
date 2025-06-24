import json
try:
    import orjson  # type: ignore
    def dumps(o):
        return orjson.dumps(o, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode()
    backend = lambda: "orjson"
except ModuleNotFoundError:
    dumps = lambda o: json.dumps(o, separators=(",", ":"))
    backend = lambda: "stdlib"
