import copy
from pydrake.all import LeafSystem, ContactResults


class ContactLogger(LeafSystem):
    def __init__(self, value, publish_period_seconds: float):
        super().__init__()
        self.DeclarePeriodicPublish(publish_period_seconds)
        self.input_port = self.DeclareAbstractInputPort(
            "value", ContactResults.Make(value)
        )
        self.sample_times = []
        self.data = []

    def DoPublish(self, context, event):
        super().DoPublish(context, event)
        self.sample_times.append(context.get_time())
        self.data.append(copy.deepcopy(self.input_port.Eval(context)))
