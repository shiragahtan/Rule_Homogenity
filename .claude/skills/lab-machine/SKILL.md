---
name: lab-machine
description: Run, monitor and collect heavy experiments of this project on the lab machine (ssh + tmux + .venv). Use whenever a run is heavy (exact checks, candidate search, RW comparison, full experiments, runtimes) or the user asks to run something "on the machine", check if a run finished, or copy results back.
---

# Lab machine

Heavy runs go on the lab machine, never on the laptop (the user's rule). Runtimes in the paper are measured there.

## Connection
- The host and user name are NOT stored in this repo (it is public). Ask the user, or use the one given earlier in the session.
- The machine is reachable only from the university network or VPN. If `ssh` times out (`Operation timed out`, ping
  100% loss) while the internet works, ask the user to connect the university VPN; you cannot do it for them.
- Passwordless: the laptop key must be installed once by the user, in a normal Terminal window:
  `ssh-copy-id -o PreferredAuthentications=password -o PubkeyAuthentication=no -i ~/.ssh/id_ed25519.pub <user>@<host>`
  Then always pass the key explicitly: `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <user>@<host> '...'`
  (without `-i` the login is refused). Never ask for or store the password.
- In zsh, do not put ssh options in a variable (`$O`): it is not word-split. Write them out.

## On the machine
- Repo: `~/Rule_Homogenity`. Always `cd ~/Rule_Homogenity && source .venv/bin/activate` first (there is no
  `python` outside the venv). 48 cores; use `--workers 32`–`40`.
- Long runs go in a detached tmux session, logging to `logs/` (git-ignored):
  `tmux new -d -s <name> "source .venv/bin/activate && python -u <script> ... 2>&1 | tee logs/<name>.log"`
- Never run two timed experiments at the same time (runtimes would be wrong). To queue, a small script can wait:
  `while tmux has-session -t <previous> 2>/dev/null; do sleep 60; done`.
- Do not kill the old sessions (`so_exact`, `eqwidth`) without asking.

## Getting files there and back
- Code: commit + push (with the user's approval), then `git pull origin dowhy-cate-estimator` on the machine.
- Changed configs or rule files not yet pushed: `scp -i ~/.ssh/id_ed25519 <files> <user>@<host>:Rule_Homogenity/configs/`.
- Results back: `scp -i ~/.ssh/id_ed25519 '<user>@<host>:~/Rule_Homogenity/<path>/*' <local path>/`.

## Waiting for a run
Run a background Bash loop on the laptop and get notified when it ends, e.g.
`until ssh ... 'grep -q "per strategy" ~/Rule_Homogenity/logs/x.log || ! tmux has-session -t x'; do sleep 120; done; ssh ... 'tail logs/x.log'`
End the remote command with `; true` so a `grep -c` of 0 does not look like a failure.

## If the user runs the commands themselves
Give only paste-ready commands, no English text between them, each including `source .venv/bin/activate`.
