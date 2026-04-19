FROM sophgo/tpuc_dev:v3.4

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    git-lfs \
    jq \
    ripgrep \
    tree \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" \
    && mkdir -p /etc/sudoers.d \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

COPY .bashrc /home/${USERNAME}/.bashrc
COPY .zshrc /home/${USERNAME}/.zshrc
COPY setup.sh /setup.sh

RUN chmod +x /setup.sh \
    && mkdir -p /home/${USERNAME}/.cache/uv \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME} /setup.sh \
    && git config --system --add safe.directory /workspace

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV HOME=/home/${USERNAME}

USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]
