# Multi-agent safety filter learning

A safety filter projects a nominal action onto the set of safe actions by
solving a QP whose constraints encode safety. When each of several agents
runs its own filter, the filter parameters can be learned jointly so that
the *team* behaviour is safe and performant.

`qpax` makes this end-to-end: batch the per-agent QPs with `jax.vmap`,
differentiate through the projected actions, and update shared safety
parameters with any JAX-compatible optimizer.

```python
import qpax  # todo: multi-agent safety filter learning example
```
