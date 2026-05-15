# MPC autotuning

A model predictive controller is a QP solved once per control step. Its
behaviour is shaped by hand-picked weight matrices — exactly the kind of
hyper-parameters that benefit from being learned end-to-end against a
downstream performance metric.

With `qpax` the MPC QP becomes a differentiable layer: roll out the
controlled system, evaluate a task loss on the rollout, and backpropagate
into the cost weights.

```python
import qpax  # todo: MPC autotuning example
```
