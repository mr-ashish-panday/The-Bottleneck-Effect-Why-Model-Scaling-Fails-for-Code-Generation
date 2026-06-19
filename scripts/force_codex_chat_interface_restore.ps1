$ErrorActionPreference = "Stop"

$CodexRoot = Join-Path $env:USERPROFILE ".codex"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupDir = Join-Path $CodexRoot "backups\force-interface-restore-$Timestamp"
$LogPath = Join-Path $CodexRoot "backups\force-interface-restore-$Timestamp.log"

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Log($Message) {
    $Message | Tee-Object -FilePath $LogPath -Append
}

Log "Force Codex chat interface restore"
Log "Started: $(Get-Date -Format o)"
Log "CodexRoot: $CodexRoot"
Log "BackupDir: $BackupDir"
Log ""

if (!(Test-Path -LiteralPath $CodexRoot)) {
    throw "Codex root not found: $CodexRoot"
}

Log "Stopping Codex processes..."
for ($i = 0; $i -lt 8; $i++) {
    Get-Process Codex,codex -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    cmd.exe /c "taskkill /F /T /IM Codex.exe >nul 2>nul"
    cmd.exe /c "taskkill /F /T /IM codex.exe >nul 2>nul"
    Start-Sleep -Seconds 2
}

$left = Get-Process Codex,codex -ErrorAction SilentlyContinue
if ($left) {
    $left | Format-Table Id,ProcessName,Path | Out-String | Tee-Object -FilePath $LogPath -Append
    throw "Codex is still running. Keep PowerShell as Administrator and rerun."
}

Log "Backing up Codex state..."
$backupFiles = @(
    ".codex-global-state.json",
    ".codex-global-state.json.bak",
    "state_5.sqlite",
    "state_5.sqlite-wal",
    "state_5.sqlite-shm",
    "session_index.jsonl"
)

foreach ($name in $backupFiles) {
    $src = Join-Path $CodexRoot $name
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $BackupDir $name) -Force
    }
}

$env:CODEX_FORCE_RESTORE_ROOT = $CodexRoot
$env:CODEX_FORCE_RESTORE_LOG = $LogPath
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path -LiteralPath $BundledPython) {
    $PythonExe = $BundledPython
} else {
    $PythonExe = "python"
}

@'
import datetime
import json
import os
import pathlib
import re
import sqlite3

root = pathlib.Path(os.environ["CODEX_FORCE_RESTORE_ROOT"])
global_path = root / ".codex-global-state.json"
db_path = root / "state_5.sqlite"
index_path = root / "session_index.jsonl"
scan_roots = [root / "sessions", root / "archived_sessions"]
uuid_re = re.compile(r"(019[0-9a-f-]+)", re.I)

def normalize_path(value):
    if isinstance(value, str) and value.startswith("\\\\?\\"):
        return value[4:]
    return value or ""

def long_variant(value):
    if isinstance(value, str) and re.match(r"^[A-Za-z]:\\", value):
        return "\\\\?\\" + value
    return None

def add_unique(items, value, raw=False):
    if not raw:
        value = normalize_path(value)
    if value and value not in items:
        items.append(value)

def clean_text(value, fallback="Untitled chat"):
    if value is None:
        value = fallback
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    return text[:160] if text else fallback

def iso_to_ms(value):
    if not value:
        return 0
    try:
        return int(datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return 0

def ms_to_iso(ms):
    try:
        return datetime.datetime.fromtimestamp(int(ms) / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
    except Exception:
        return "1970-01-01T00:00:00.0000000Z"

def extract_rollout(path):
    meta = {}
    turn = {}
    first_user = ""
    last_ts = ""
    tokens = 0
    has_user = False
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            item = json.loads(line)
        except Exception:
            continue
        ts = item.get("timestamp") or ""
        if ts:
            last_ts = ts
        if item.get("type") == "session_meta" and not meta:
            meta = item.get("payload") or {}
        elif item.get("type") == "turn_context" and not turn:
            turn = item.get("payload") or {}
        elif item.get("type") == "response_item" and (item.get("payload") or {}).get("type") == "message":
            payload = item.get("payload") or {}
            if payload.get("role") == "user":
                parts = []
                for c in payload.get("content") or []:
                    if isinstance(c, dict):
                        parts.append(c.get("text") or c.get("input_text") or "")
                msg = "\n".join(parts).strip()
                if msg and not msg.startswith("# AGENTS.md") and not msg.startswith("<environment_context>") and not msg.startswith("<turn_aborted>"):
                    has_user = True
                    if not first_user:
                        first_user = msg
        elif item.get("type") == "event_msg" and (item.get("payload") or {}).get("type") == "token_count":
            try:
                tokens = int(item["payload"]["info"]["total_token_usage"]["total_tokens"])
            except Exception:
                pass
    return meta, turn, first_user, last_ts, tokens, has_user

existing_index = {}
existing_records = []
if index_path.exists():
    for line in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        tid = item.get("id")
        if tid:
            existing_records.append(item)
            if tid not in existing_index or (item.get("updated_at") or "") >= (existing_index[tid].get("updated_at") or ""):
                existing_index[tid] = item

rollouts = {}
for scan_root in scan_roots:
    if not scan_root.exists():
        continue
    for path in scan_root.rglob("rollout-*.jsonl"):
        match = uuid_re.search(path.name)
        if match:
            rollouts[match.group(1)] = path

rollout_meta = {}
for tid, path in rollouts.items():
    rollout_meta[tid] = extract_rollout(path)

con = sqlite3.connect(str(db_path), timeout=30)
con.row_factory = sqlite3.Row
cur = con.cursor()

existing_db_ids = {row["id"] for row in cur.execute("select id from threads")}
inserted = 0
for tid, path in sorted(rollouts.items()):
    if tid in existing_db_ids:
        continue
    meta, turn, first_user, last_ts, tokens, has_user = rollout_meta[tid]
    created_ms = iso_to_ms(meta.get("timestamp")) or iso_to_ms(last_ts) or int(path.stat().st_mtime * 1000)
    updated_ms = iso_to_ms(last_ts) or int(path.stat().st_mtime * 1000)
    title = clean_text((existing_index.get(tid) or {}).get("thread_name") or first_user or path.stem)
    cwd = normalize_path(meta.get("cwd") or turn.get("cwd") or str(path.parent))
    sandbox_policy = json.dumps(turn.get("sandbox_policy") or {"type": "danger-full-access"}, ensure_ascii=False, separators=(",", ":"))
    cur.execute(
        """
        insert into threads (
            id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
            sandbox_policy, approval_mode, tokens_used, has_user_event, archived, archived_at,
            git_sha, git_branch, git_origin_url, cli_version, first_user_message, agent_nickname,
            agent_role, memory_mode, model, reasoning_effort, agent_path, created_at_ms,
            updated_at_ms, thread_source
        ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            tid, str(path), created_ms // 1000, updated_ms // 1000,
            meta.get("source") if isinstance(meta.get("source"), str) else "vscode",
            meta.get("model_provider") if isinstance(meta.get("model_provider"), str) else "openai",
            cwd, title, sandbox_policy, turn.get("approval_policy") or "never", int(tokens or 0),
            1 if has_user else 0, 0, None, None, None, None, meta.get("cli_version") or "",
            first_user or title, None, None, turn.get("memory_mode") or "enabled",
            turn.get("model"), turn.get("effort") or turn.get("reasoning_effort"), None,
            created_ms, updated_ms, "user" if has_user else None,
        ),
    )
    inserted += 1

changed_cwd = 0
changed_rollout = 0
marked_user_event = 0
marked_thread_source = 0
filled_first = 0
filled_title = 0
thread_roots = []

for row in cur.execute("select * from threads").fetchall():
    tid = row["id"]
    new_cwd = normalize_path(row["cwd"])
    new_rollout = normalize_path(row["rollout_path"])
    add_unique(thread_roots, new_cwd)
    meta = rollout_meta.get(tid)
    first_user = meta[2] if meta else ""
    has_user = bool(meta[5]) if meta else bool(row["first_user_message"])
    title = row["title"] or ""
    first = row["first_user_message"] or ""
    source = row["source"] or ""
    is_subagent = "subagent" in source or bool(row["agent_role"])
    new_has_user_event = row["has_user_event"]
    new_thread_source = row["thread_source"]
    if not row["archived"] and has_user:
        new_has_user_event = 1
        if not is_subagent:
            new_thread_source = "user"
    if first_user and not first:
        first = first_user
    if not title:
        title = clean_text(first or (existing_index.get(tid) or {}).get("thread_name") or tid)
    if (
        new_cwd != row["cwd"] or new_rollout != row["rollout_path"] or
        new_has_user_event != row["has_user_event"] or new_thread_source != row["thread_source"] or
        first != (row["first_user_message"] or "") or title != (row["title"] or "")
    ):
        cur.execute(
            """
            update threads
            set cwd=?, rollout_path=?, has_user_event=?, thread_source=?, first_user_message=?, title=?
            where id=?
            """,
            (new_cwd, new_rollout, new_has_user_event, new_thread_source, first, title, tid),
        )
        changed_cwd += int(new_cwd != row["cwd"])
        changed_rollout += int(new_rollout != row["rollout_path"])
        marked_user_event += int(new_has_user_event != row["has_user_event"])
        marked_thread_source += int(new_thread_source != row["thread_source"])
        filled_first += int(first != (row["first_user_message"] or ""))
        filled_title += int(title != (row["title"] or ""))

con.commit()

data = json.loads(global_path.read_text(encoding="utf-8"))
roots = []
for key in ("active-workspace-roots", "electron-saved-workspace-roots", "project-order"):
    for value in data.get(key) or []:
        add_unique(roots, value)
for value in thread_roots:
    add_unique(roots, value)
for value in list(roots):
    add_unique(roots, long_variant(value), raw=True)
data["active-workspace-roots"] = roots
data["electron-saved-workspace-roots"] = roots
data["project-order"] = roots
eps = data.setdefault("electron-persisted-atom-state", {})
collapsed_sections = eps.setdefault("sidebar-collapsed-sections-v1", {})
collapsed_sections["chats"] = False
collapsed_sections["pinned"] = False
collapsed_sections["threads"] = False
collapsed_groups = eps.setdefault("sidebar-collapsed-groups", {})
for key in list(collapsed_groups.keys()):
    collapsed_groups[key] = False
data["thread-titles"] = data.get("thread-titles") or {"titles": {}, "order": []}
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
global_path.write_text(payload, encoding="utf-8")
(root / ".codex-global-state.json.bak").write_text(payload, encoding="utf-8")

db_rows = cur.execute("select id,title,updated_at_ms,updated_at from threads").fetchall()
entries = {}
for row in db_rows:
    existing = existing_index.get(row["id"]) or {}
    entries[row["id"]] = {
        "id": row["id"],
        "thread_name": existing.get("thread_name") or clean_text(row["title"]),
        "updated_at": existing.get("updated_at") or ms_to_iso(row["updated_at_ms"] or row["updated_at"] * 1000),
    }
for tid, path in rollouts.items():
    if tid not in entries:
        existing = existing_index.get(tid) or {}
        entries[tid] = {
            "id": tid,
            "thread_name": existing.get("thread_name") or clean_text(path.stem),
            "updated_at": existing.get("updated_at") or ms_to_iso(path.stat().st_mtime * 1000),
        }
orphan_count = 0
for item in existing_records:
    tid = item.get("id")
    if tid and tid not in entries:
        entries[tid] = {
            "id": tid,
            "thread_name": item.get("thread_name") or "Untitled chat",
            "updated_at": item.get("updated_at") or "1970-01-01T00:00:00.0000000Z",
        }
        orphan_count += 1
ordered = sorted(entries.values(), key=lambda item: item.get("updated_at") or "")
index_path.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in ordered), encoding="utf-8")

integrity = cur.execute("pragma integrity_check").fetchone()[0]
threads_total = cur.execute("select count(*) from threads").fetchone()[0]
threads_unarchived = cur.execute("select count(*) from threads where archived=0").fetchone()[0]
has_user_event_visible = cur.execute("select count(*) from threads where archived=0 and has_user_event=1").fetchone()[0]
thread_source_user = cur.execute("select count(*) from threads where archived=0 and thread_source='user'").fetchone()[0]
extended_cwd = cur.execute("select count(*) from threads where cwd like '\\\\?\\%'").fetchone()[0]
extended_rollout = cur.execute("select count(*) from threads where rollout_path like '\\\\?\\%'").fetchone()[0]
missing_rollout = 0
for (rollout_path,) in cur.execute("select rollout_path from threads"):
    if rollout_path and not pathlib.Path(rollout_path).exists():
        missing_rollout += 1
visible_roots = cur.execute(
    "select count(*) from threads where archived=0 and cwd in (%s)" % ",".join("?" for _ in roots),
    tuple(roots),
).fetchone()[0]
db_ids = {row["id"] for row in db_rows}
index_ids = {item["id"] for item in ordered}
try:
    cur.execute("pragma wal_checkpoint(full)")
except Exception:
    pass
con.close()

print(f"inserted_missing_db_rows={inserted}")
print(f"changed_cwd={changed_cwd}")
print(f"changed_rollout={changed_rollout}")
print(f"marked_user_event={marked_user_event}")
print(f"marked_thread_source={marked_thread_source}")
print(f"filled_first_user_message={filled_first}")
print(f"filled_title={filled_title}")
print(f"integrity={integrity}")
print(f"threads_total={threads_total}")
print(f"threads_unarchived={threads_unarchived}")
print(f"has_user_event_visible={has_user_event_visible}")
print(f"thread_source_user={thread_source_user}")
print(f"extended_cwd={extended_cwd}")
print(f"extended_rollout={extended_rollout}")
print(f"visible_active_roots={visible_roots}")
print(f"active_workspace_roots={len(roots)}")
print(f"missing_rollout_files={missing_rollout}")
print(f"session_index_rows={len(ordered)}")
print(f"session_index_unique={len(index_ids)}")
print(f"session_index_orphan_rows_preserved={orphan_count}")
print(f"session_index_db_missing={len(db_ids - index_ids)}")

if integrity != "ok":
    raise SystemExit("SQLite integrity failed")
if missing_rollout:
    raise SystemExit("Missing rollout files remain")
if db_ids - index_ids:
    raise SystemExit("DB threads missing from session index")
'@ | & $PythonExe -

if ($LASTEXITCODE -ne 0) {
    throw "Force restore failed. Backup remains at: $BackupDir"
}

Log ""
Log "Launching Codex..."
$codexCommand = Get-Command codex -ErrorAction SilentlyContinue
if ($codexCommand) {
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $codexCommand.Source,
        "app"
    ) -WindowStyle Hidden
} else {
    Log "Codex CLI not found on PATH. Open Codex Desktop manually."
}

Log "Finished: $(Get-Date -Format o)"
Log "LogPath: $LogPath"
