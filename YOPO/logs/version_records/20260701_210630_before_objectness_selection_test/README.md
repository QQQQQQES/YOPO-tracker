# Version record before objectness/hybrid selection test

Time: 20260701_210630
Repo: /mnt/nvme0n1p5/YOPO
Working dir: /mnt/nvme0n1p5/YOPO/YOPO
Purpose: before testing YOPOv2 runtime selection using objectness/hybrid instead of pure score=min.

Important context:
- No code change is required for first test because test_yopo_ros.py already has --selection_mode score|objectness|hybrid.
- This record captures current file snapshot and diff so the runtime file can be restored manually if needed.

Files:
- git_head.txt
- git_status_short.txt
- test_yopo_ros.diff
- test_yopo_ros.py.snapshot
