
class StageBase:

    state_space_class = None
    parameter_class = None

    def __init__(self, args) -> None:
        super().__init__()
        self.state_space = self.state_space_class(args)

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def run(self):
        for parameter_key in self.state_space:
            parameter = self.parameter_class(parameter_key)
            parameter.try_evaluate()

    def merge(self):
        data = []
        for parameter_key in self.state_space:
            parameter = self.parameter_class(parameter_key)
            result = parameter.load()
            if result is None:
                return data
            data.append(result)
        return data
