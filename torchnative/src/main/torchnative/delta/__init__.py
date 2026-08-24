"""Weight deltas over base weights.

Every adaptation method produces one of these; they differ only in lifetime
and destination. The boundary is ``model(x)`` -- state a model manages inside
its own forward (fast weights, caches) is the model's, not ours.

Lifetime names are deliberately unnamed for now. They are driven by system
events -- backgrounding, user switch, sync window -- not by the domain
boundaries a benchmark hands you, and they get chosen once the first
integration shows the real usage. See DESIGN.md §3.
"""
