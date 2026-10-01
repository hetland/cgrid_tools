# Examples

Install the package from the repository root with `python -m pip install -e .`,
then run any script directly, or run all three with `make examples`.

- `simple_focused_channel.py`: all three four-side solvers and grid-line focus.
- `estuary.py`: a segmented four-sided boundary and off-center focus.
- `side_channel.py`: a reentrant domain using CRDT and its exclusion mask.

Each script checks basic output quality and returns a nonzero exit status on
failure. See `docs/examples/` for executable notebooks with reference plots.
