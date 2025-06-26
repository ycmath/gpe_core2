from gpe_core2.orchestrator import Orchestrator
import time, pprint

o = Orchestrator()
t0, p1 = time.time(), o.process_query("popular fruits"); t1 = time.time()
t2, p2 = time.time(), o.process_query("popular fruits"); t3 = time.time()

print(f"1st call: {(t1-t0):.4f}s  cached? {p1.metadata['cached']}")
print(f"2nd call: {(t3-t2):.4f}s  cached? {p2.metadata['cached']}")
pprint.pp(p2.generative_payload)
