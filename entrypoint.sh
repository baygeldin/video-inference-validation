#!/usr/bin/env bash
set -euo pipefail

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

    /usr/sbin/sshd -D -e &
}

start_ssh

exec "$@"
