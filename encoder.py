from dataclasses import asdict

class GPEEncoder:
    def __init__(self, include_fallback: bool = True):
        self.include_fallback = include_fallback

    def encode(self, data):
        builder = ASTBuilder()
        root_id = builder.build(data)
        rep = RepetitionDetector(builder.nodes).detect()
        seeds = SeedGenerator(builder.nodes, rep).generate()
        gen = {
            "version": "gpe.v1",
            "root_id": root_id,
            "seeds": [asdict(s) for s in seeds],
        }
        fb = dumps(data) if self.include_fallback else None
        return GpePayload(
            payload_type="gpe.v1",
            generative_payload=gen,
            fallback_payload={"json": fb} if fb else None,
        )
