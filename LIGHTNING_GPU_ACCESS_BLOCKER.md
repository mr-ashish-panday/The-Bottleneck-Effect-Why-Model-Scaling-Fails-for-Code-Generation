# Lightning GPU Access Blocker

Current blocker: SSH authentication reaches `ssh.lightning.ai`, but Lightning rejects the local public key.

Observed command:

```bash
ssh -i ~/.ssh/lightning_rsa -o IdentitiesOnly=yes s_01kv4xj262ce96tjdepmyc708w@ssh.lightning.ai
```

Observed error:

```text
Permission denied (publickey).
```

## Public Key To Add In Lightning

Fingerprint:

```text
SHA256:LQRMu0661WMuB/F/QTKFUujxAa75oBfoJiNLQCxobyM
```

Public key:

```text
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC6hyZnsyRs5Op9YRYH53YQLoccqGJmvqJW/53eqMoIDZET6N6OhoKg1VvEL4BCUDyTqG2UYtLCan0H9+SlUVwVBo8ZmOy3Nf9Kf1/YE5u/AFs/paVdAdB0bFDrOy1wf/pgNukaksOYnoLxLwsnNINhzXoULHE74xDxIr0On5HGmTXae/5BYikmxURyuL8xmNiz/6WSX2EaFW3jKOM/gFQGVP4Z8CSiTsUDwGFdgpif5CU0jJjRLSPNMXBmU/3qaTMMkOkeHaHm0POoMsWfja6yqygFx8bKyM2koxOSKg/ftf+47cGfMahCGN9/G39b9q1TtK8mXFEjBuG0ySdSNCVV
```

## After Adding The Key

Run this from the local project root:

```powershell
ssh -i $env:USERPROFILE\.ssh\lightning_rsa -o IdentitiesOnly=yes s_01kv4xj262ce96tjdepmyc708w@ssh.lightning.ai "pwd; hostname; nvidia-smi"
```

If that succeeds, upload and start:

```powershell
scp -i $env:USERPROFILE\.ssh\lightning_rsa .\trump-occurrence-gpu-transfer-2026-06-15.zip s_01kv4xj262ce96tjdepmyc708w@ssh.lightning.ai:~/trump-occurrence-gpu-transfer-2026-06-15.zip
ssh -i $env:USERPROFILE\.ssh\lightning_rsa s_01kv4xj262ce96tjdepmyc708w@ssh.lightning.ai "mkdir -p ~/trump-occurrence-work && unzip -o ~/trump-occurrence-gpu-transfer-2026-06-15.zip -d ~/trump-occurrence-work && cd ~/trump-occurrence-work/trump-occurrence-gpu-upgrade && bash scripts/setup_remote.sh && source .venv/bin/activate && python -m src.inspect_gpu"
```

