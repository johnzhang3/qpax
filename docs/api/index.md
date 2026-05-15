# API reference

Every public symbol is exported from the top-level `qpax` namespace and
auto-rendered from its docstring. The reference is split into three layers:

- **[Core API](core.md)** — the dispatchers you normally import as
  `qpax.<name>`.
- **[Explicit API](explicit.md)** — internals of `backend="e"`.
- **[Implicit API](implicit.md)** — internals of `backend="i"`.

Most users only need the Core API. The backend pages are useful when you
want to call a specific implementation directly or read its docstring.
