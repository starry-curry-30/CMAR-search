"""Create empty runtime state without replacing an existing experiment."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
(root / "runtime").mkdir(exist_ok=True)
for name in (
    "all_objects.txt", "blacklist.txt", "check_find_target.txt",
    "execute_action.txt", "rgb_find_target.txt", "stuck.txt", "target_id.txt",
    "object_id.txt", "object_relations.txt", "room_seg.txt", "area.txt",
):
    (root / name).touch(exist_ok=True)
print("Runtime directory and state files are ready; existing content was preserved.")
