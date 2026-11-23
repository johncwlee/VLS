# Moving VLS from Turing to the lab desktop (written 2026-11-23, not run yet)

Why: Turing's 4 GPUs are full (root vLLM container `e6642c41`, Qwen3.8-Flash on port 8000, 44 of 49 GB per GPU,
started Sun Nov 22 07:36 UTC; owner unknown, likely johnl). Desktop: 1x RTX 4090 24 GB, driver 595.99.02, idle.
Peak GPU memory of one run (pi0.5 + 20 particles) was never measured: check in step 5.

Which machine is which: this Claude session ran on Turing (192.168.42.135). VS Code was on another machine,
probably 192.168.42.2 over Remote-SSH (assumption). If that machine is the desktop, open a local (not Remote-SSH)
VS Code window there after the copy, and Claude Code runs on the desktop directly.

Assumes the desktop username is `rafid` (same paths, so the venv, ~/.libero and Claude memory need no edits).
If not, see Fallback.

## 0. Turing, before copying (optional backup)
    cd ~/VLS_check/VLS && git push origin closed_loop
Puts the 3 unpushed commits (ahead of origin/closed_loop) on John's fork. Uncommitted work moves only through the copy.
After step 2, stop editing the Turing copy: one working copy only.

## 1. Desktop: packages and checks
    whoami                      # expect: rafid (if not: Fallback)
    df -h ~                     # need ~30 GB free
    sudo apt install -y rsync git git-lfs libegl1 libgl1
    git lfs install             # third_party/lerobot uses LFS; Turing sets filter.lfs.required=true

## 2. Desktop: copy (~25 GB; re-run the same lines to resume)
    S=rafid@192.168.42.135
    mkdir -p ~/VLS_check ~/.local/share/uv ~/.local/bin ~/.cache ~/.claude/projects
    rsync -aH --info=progress2 $S:VLS_check/{VLS,.venv-vls,HANDOFF.md,RESULTS_2026-11-08.md,venv_freeze_2026-11-23.txt} ~/VLS_check/
    rsync -aH $S:.local/share/uv/python ~/.local/share/uv/
    rsync -a  $S:.local/bin/uv ~/.local/bin/
    rsync -aH --info=progress2 $S:.cache/huggingface ~/.cache/
    rsync -a  $S:.cache/torch ~/.cache/
    rsync -a  $S:.libero ~/
    rsync -a  $S:.gitconfig ~/
    rsync -a  $S:.claude/{CLAUDE.md,settings.json,skills,plugins} ~/.claude/
    rsync -a  $S:.claude/projects/-home-rafid-VLS-check-VLS ~/.claude/projects/

What each piece is:
- VLS (2.5 GB): code, .git, uncommitted work, outputs/ (892 MB), notes/, LOG/TODO/CLAUDE.md, third_party/ (LIBERO-PRO assets inside).
- .venv-vls (8.5 GB, uv, Python 3.12.15, torch 2.7.1+cu118) points to ~/.local/share/uv/python (223 MB).
- ~/.cache/huggingface (15 GB): pi0.5 weights live in the shared hub/blobs folder, so copy all of it with symlinks (-a).
- ~/.cache/torch: dinov2_vits14. ~/.libero/config.yaml: absolute LIBERO paths.

## 3. Desktop: logins
    claude                      # log in; closed-loop VLM calls use this login (claude_agent_sdk)
    ssh-keygen -t ed25519       # if no key yet
    cat ~/.ssh/id_ed25519.pub   # add at github.com -> Settings -> SSH keys
    ssh -T git@github.com       # expect "Hi rsaif22!"

## 4. Desktop: checks
    cd ~/VLS_check/VLS && source ../.venv-vls/bin/activate && export MUJOCO_GL=egl HF_HUB_OFFLINE=1
    git status --short          # slow once (index rebuild); same changes as on Turing
    git log --oneline -1        # expect 9e90b45
    git fetch origin
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    python -c "import mujoco, robosuite, libero, lerobot; print('imports ok')"
    python -m pytest tests -q
Expected pytest on 2026-11-23 Turing: 10 failures in tests/test_closed_loop_run.py (FakeVLM signature, see
notes/handoff.md), the rest pass. Run with a GPU visible: with CUDA_VISIBLE_DEVICES="" the EGL test errors.

## 5. Desktop: sim smoke test (task 0, init state 0, old position-only behavior)
Second terminal: `watch -n 2 nvidia-smi`, note the peak. Then:

    CUDA_VISIBLE_DEVICES=0 python scratch/2026-11-19_fixed_init_launcher.py backend=libero \
      backend.libero.suite_name=libero_goal_task main.episode_num=1 'backend.libero.task_ids_filter=[0]' \
      main.use_fkd=false perception.keypoint_detector.feature_extractor=dinov2_vits14 main.use_guidance=true \
      main.sample_batch_size=20 main.log_steps=true main.closed_loop_vls=true experiment.name=port_smoke_t0

On CUDA out-of-memory: stop and report. Do not lower sample_batch_size (changes steering strength, CLAUDE.md gotcha).

## 6. Continue
Open ~/VLS_check/VLS in VS Code on the desktop, run /session-start, then follow notes/handoff.md.

## Fallback: username is not `rafid`
Steps 1-3 without `.venv-vls` and `.local/share/uv/python` in step 2, then:

    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd ~/VLS_check && uv venv .venv-vls --python 3.12.15 && source .venv-vls/bin/activate
    grep -v -E '^-e |@ file://' venv_freeze_2026-11-23.txt > /tmp/req.txt
    uv pip install -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
    uv pip install -e VLS/third_party/lerobot -e VLS/third_party/libero_pro -e VLS/third_party/sam3 VLS/third_party/transformers
    rsync -a rafid@192.168.42.135:VLS_check/.venv-vls/lib/python3.12/site-packages/robosuite/macros_private.py \
      .venv-vls/lib/python3.12/site-packages/robosuite/
    sed -i "s|/home/rafid|$HOME|g" ~/.libero/config.yaml
    mv ~/.claude/projects/-home-rafid-VLS-check-VLS ~/.claude/projects/$(echo $HOME/VLS_check/VLS | tr / -)
Then step 4.

## Alternative: drive the desktop from Turing over SSH
Desktop: `sudo apt install -y openssh-server`, `hostname -I`. Turing: `ssh-keygen -t ed25519`, add the public key
to the desktop's ~/.ssh/authorized_keys. Then a Turing session can run the steps with `ssh rafid@<desktop-ip>`.
