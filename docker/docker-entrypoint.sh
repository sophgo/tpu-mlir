#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}

echo "Starting with UID : $USER_ID"
useradd --shell /bin/bash -u $USER_ID -o -c "" -m user
mkdir -p /etc/sudoers.d
echo 'user ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/user
chmod 440 /etc/sudoers.d/user

export HOME=/home/user

exec gosu user "$@"

