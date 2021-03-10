from base.utils import insert_layer_nonseq
from dropin.layers import DropinProfiler


class Dropin:

    def __init__(self, model, representative_data) -> None:
        super().__init__()
        self.model = model
        self.representative_data = representative_data

        def profiler_layer_factory(insert_layer_name):
            return DropinProfiler(name=insert_layer_name)
        profiler = insert_layer_nonseq(model, 'conv2d.*|dense.*', profiler_layer_factory, 'profiler')
        profiler.predict(representative_data)
        self.a, self.b = DropinProfiler.a, DropinProfiler.b
