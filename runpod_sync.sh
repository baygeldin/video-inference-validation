#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  runpod-sync [pull] HOST PORT [--dry-run] [--delete]

Pull remote /workspace/ into this pod's /workspace/.

Options:
  --dry-run   Show what would change without copying files.
  --delete    Delete local files that do not exist on the remote pod.
  -h, --help  Show this help.

Environment overrides:
  SYNC_KEY_PATH    SSH private key path. Default: /root/.ssh/runpod_sync
  SYNC_USER        SSH user. Default: root
  SYNC_SOURCE_DIR  Remote source directory. Default: /workspace
  SYNC_DEST_DIR    Local destination directory. Default: /workspace
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "pull" ]]; then
    shift
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

host="$1"
port="$2"
shift 2

dry_run=0
delete_extra=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --delete)
            delete_extra=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

sync_key_path="${SYNC_KEY_PATH:-/root/.ssh/runpod_sync}"
sync_user="${SYNC_USER:-root}"
source_dir="${SYNC_SOURCE_DIR:-/workspace}"
dest_dir="${SYNC_DEST_DIR:-/workspace}"

if [[ ! -f "${sync_key_path}" ]]; then
    cat >&2 <<EOF
Missing sync key at ${sync_key_path}.
Set SYNC_PRIVATE_KEY on the pod so entrypoint.sh can create it at startup.
EOF
    exit 1
fi

chmod 600 "${sync_key_path}"
mkdir -p "${dest_dir}"

rsync_args=(
    -aH
    --partial
    --info=progress2
)

if [[ "${dry_run}" -eq 1 ]]; then
    rsync_args+=(--dry-run --itemize-changes)
fi

if [[ "${delete_extra}" -eq 1 ]]; then
    rsync_args+=(--delete)
fi

remote="${sync_user}@${host}:${source_dir%/}/"
destination="${dest_dir%/}/"
ssh_command="ssh -i ${sync_key_path} -p ${port} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/.ssh/known_hosts"

printf 'Pulling %s -> %s\n' "${remote}" "${destination}"
rsync "${rsync_args[@]}" -e "${ssh_command}" "${remote}" "${destination}"
