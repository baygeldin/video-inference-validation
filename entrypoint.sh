#!/usr/bin/env bash
set -euo pipefail

configure_sync_key() {
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh

    if [[ -z "${SYNC_PRIVATE_KEY:-}" ]]; then
        return
    fi

    local sync_key_path="/root/.ssh/runpod_sync"
    printf '%s\n' "${SYNC_PRIVATE_KEY}" | sed 's/\\n/\n/g' > "${sync_key_path}"
    chmod 600 "${sync_key_path}"

    local sync_public_key
    sync_public_key="$(ssh-keygen -y -f "${sync_key_path}")"
    touch /root/.ssh/authorized_keys
    grep -qxF "${sync_public_key}" /root/.ssh/authorized_keys || \
        printf '%s\n' "${sync_public_key}" >> /root/.ssh/authorized_keys
}

# See: https://docs.runpod.io/pods/configuration/use-ssh
start_ssh() {
    mkdir -p /run/sshd /root/.ssh
    chmod 700 /root/.ssh

    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        printf '%s\n' "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    fi

    if [[ -f /root/.ssh/authorized_keys ]]; then
        chmod 600 /root/.ssh/authorized_keys
    fi

    ssh-keygen -A
    /usr/sbin/sshd -D -e &
}

configure_sync_key
start_ssh

exec "$@"
