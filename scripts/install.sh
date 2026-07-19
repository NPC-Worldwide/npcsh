#!/bin/sh
# Install the latest npcsh Rust binaries.
# Usage: curl -fsSL https://enpisi.com/install-npcsh.sh | sh

set -e

REPO="NPC-Worldwide/npcsh"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.npcsh/bin}"

get_os() {
    case "$(uname -s)" in
        Linux*) echo "linux" ;;
        Darwin*) echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

get_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        arm64|aarch64) echo "aarch64" ;;
        *) echo "unknown" ;;
    esac
}

OS="$(get_os)"
ARCH="$(get_arch)"

if [ "$OS" = "unknown" ] || [ "$ARCH" = "unknown" ]; then
    echo "Unsupported platform: $(uname -s) $(uname -m)" >&2
    exit 1
fi

EXT=""
if [ "$OS" = "windows" ]; then
    EXT=".exe"
fi

fetch_latest_tag() {
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name":' \
        | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/'
}

TAG="${TAG:-$(fetch_latest_tag)}"
if [ -z "$TAG" ]; then
    echo "Could not determine latest release." >&2
    exit 1
fi

echo "Installing npcsh ${TAG} for ${OS}/${ARCH}..."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$INSTALL_DIR"

install_binary() {
    name="$1"
    asset="${name}-${OS}-${ARCH}${EXT}"
    url="https://github.com/${REPO}/releases/download/${TAG}/${asset}"
    tmp_bin="${TMP_DIR}/${name}${EXT}"
    install_path="${INSTALL_DIR}/${name}${EXT}"

    echo "  downloading ${asset}..."
    curl -fsSL "$url" -o "$tmp_bin"
    chmod +x "$tmp_bin"
    cp "$tmp_bin" "$install_path"
    echo "  installed ${install_path}"
}

install_binary "npcsh"
install_binary "npc"

# macOS: if a downloaded binary is not signed, apply an ad-hoc signature so
# Gatekeeper does not kill it on first run. Signed release binaries skip this.
if [ "$OS" = "macos" ] && command -v codesign >/dev/null 2>&1; then
    for bin in "${INSTALL_DIR}/npcsh" "${INSTALL_DIR}/npc"; do
        if ! codesign -v "$bin" >/dev/null 2>&1; then
            echo "  ad-hoc signing ${bin} for macOS..."
            codesign -s - -f "$bin" >/dev/null
        fi
    done
fi

# Add ~/.npcsh/bin to PATH in the user's shell rc file (idempotent).
case ":${PATH}:" in
    *":${INSTALL_DIR}:") ;;
    *)
        SHELL_NAME="$(basename "${SHELL:-sh}")"
        case "$SHELL_NAME" in
            zsh) RC_FILE="$HOME/.zshrc" ;;
            bash)
                if [ "$OS" = "macos" ] && [ -f "$HOME/.bash_profile" ]; then
                    RC_FILE="$HOME/.bash_profile"
                else
                    RC_FILE="$HOME/.bashrc"
                fi
                ;;
            *) RC_FILE="$HOME/.profile" ;;
        esac

        if [ -f "$RC_FILE" ] && grep -q 'npcsh/bin' "$RC_FILE"; then
            echo "  PATH for npcsh already configured in ${RC_FILE}"
        else
            touch "$RC_FILE"
            printf '\n# npcsh binaries\nexport PATH="%s:$PATH"\n' "$INSTALL_DIR" >> "$RC_FILE"
            echo "  added ${INSTALL_DIR} to PATH in ${RC_FILE}"
            echo "  (open a new shell, or run: . ${RC_FILE})"
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# npcpy Python backend (temporary requirement until the npcrs Rust-native
# runner lands). npcsh spawns `python3 -m npcpy.serve` on startup, so some
# Python >= 3.10 must have npcpy importable.
# ---------------------------------------------------------------------------

NPCSHRC="$HOME/.npcshrc"
VENV_DIR="$HOME/.npcsh/venv"

have_npcpy() {
    "$1" -c "import npcpy" >/dev/null 2>&1
}

pin_backend_python() {
    touch "$NPCSHRC"
    if grep -q '^export BACKEND_PYTHON_PATH=' "$NPCSHRC" 2>/dev/null; then
        TMP_RC="$(mktemp)"
        sed "s|^export BACKEND_PYTHON_PATH=.*|export BACKEND_PYTHON_PATH=$1|" "$NPCSHRC" > "$TMP_RC"
        mv "$TMP_RC" "$NPCSHRC"
    else
        printf 'export BACKEND_PYTHON_PATH=%s\n' "$1" >> "$NPCSHRC"
    fi
    echo "  pinned BACKEND_PYTHON_PATH=$1 in ~/.npcshrc"
}

echo ""
echo "Setting up the npcpy Python backend..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "  WARNING: python3 not found. Install Python 3.10+, then run: python3 -m pip install npcpy"
elif have_npcpy python3; then
    echo "  npcpy is already installed for $(command -v python3)."
else
    PYVER_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)' 2>/dev/null || echo 0)"
    if [ "$PYVER_OK" != "1" ]; then
        echo "  WARNING: $(python3 --version 2>&1) is older than 3.10; npcpy requires Python >= 3.10."
    fi

    # Build the list of install options.
    set --
    SEEN=":"
    for CAND in "$VIRTUAL_ENV" "$VENV_DIR" "./.venv" "./venv"; do
        if [ -n "$CAND" ] && [ -x "$CAND/bin/python" ]; then
            CAND_ABS="$(cd "$CAND" 2>/dev/null && pwd)"
            case "$SEEN" in
                *":${CAND_ABS}:"*) ;;
                *)
                    SEEN="${SEEN}${CAND_ABS}:"
                    set -- "$@" "venv|${CAND_ABS}"
                    ;;
            esac
        fi
    done
    if command -v uv >/dev/null 2>&1; then set -- "$@" "uv|"; fi
    if command -v pyenv >/dev/null 2>&1; then set -- "$@" "pyenv|"; fi
    set -- "$@" "newvenv|" "system|" "skip|"
    N=$#

    echo "  npcpy is not installed for python3. Where should it go?"
    i=1
    for OPT in "$@"; do
        case "$OPT" in
            "venv|"*)    echo "    $i) Use existing virtualenv at ${OPT#venv|}" ;;
            "uv|"*)      echo "    $i) Create a virtualenv with uv at ${VENV_DIR}" ;;
            "pyenv|"*)   echo "    $i) Install into the active pyenv Python ($(pyenv version-name 2>/dev/null || echo '?'))" ;;
            "newvenv|"*) echo "    $i) Create a virtualenv with python3 -m venv at ${VENV_DIR}" ;;
            "system|"*)  echo "    $i) Install into the system python3 (pip install --user)" ;;
            "skip|"*)    echo "    $i) Skip — I will install npcpy myself" ;;
        esac
        i=$((i + 1))
    done

    # Prompt via /dev/tty so this works under `curl | sh`. In CI or with
    # NPCSH_NONINTERACTIVE set, auto-pick: uv if available, else a fresh venv.
    CHOICE=""
    if [ -z "${CI:-}" ] && [ -z "${NPCSH_NONINTERACTIVE:-}" ] && [ -e /dev/tty ]; then
        printf "  Select an option [1]: " > /dev/tty
        read -r CHOICE < /dev/tty || CHOICE=""
        CHOICE="${CHOICE:-1}"
    else
        j=1
        for OPT in "$@"; do
            case "$OPT" in "uv|"*) CHOICE=$j; break ;; esac
            j=$((j + 1))
        done
        if [ -z "$CHOICE" ]; then
            j=1
            for OPT in "$@"; do
                case "$OPT" in "newvenv|"*) CHOICE=$j; break ;; esac
                j=$((j + 1))
            done
        fi
    fi

    OPT=""
    case "$CHOICE" in
        ''|*[!0-9]*) ;;
        *)
            j=1
            for O in "$@"; do
                if [ "$j" = "$CHOICE" ]; then OPT="$O"; fi
                j=$((j + 1))
            done
            ;;
    esac
    if [ -z "$OPT" ]; then
        echo "  invalid choice; skipping npcpy setup."
        OPT="skip|"
    fi

    PY=""
    case "$OPT" in
        "venv|"*)
            PY="${OPT#venv|}/bin/python"
            "$PY" -m pip install --quiet npcpy || true
            ;;
        "uv|"*)
            if [ ! -x "$VENV_DIR/bin/python" ]; then
                uv venv "$VENV_DIR" || true
            fi
            uv pip install --quiet --python "$VENV_DIR/bin/python" npcpy || true
            PY="$VENV_DIR/bin/python"
            ;;
        "pyenv|"*)
            PY="$(pyenv which python3 2>/dev/null)"
            if [ -n "$PY" ]; then "$PY" -m pip install --quiet npcpy || true; fi
            ;;
        "newvenv|"*)
            python3 -m venv "$VENV_DIR" || true
            PY="$VENV_DIR/bin/python"
            if [ -x "$PY" ]; then "$PY" -m pip install --quiet npcpy || true; fi
            ;;
        "system|"*)
            python3 -m pip install --quiet --user npcpy || true
            PY="python3"
            ;;
        "skip|"*)
            echo "  skipped. Install later with: python3 -m pip install npcpy"
            ;;
    esac

    if [ -n "$PY" ]; then
        if have_npcpy "$PY"; then
            echo "  npcpy installed successfully."
            if [ "$PY" != "python3" ]; then
                pin_backend_python "$PY"
            fi
        else
            echo "  WARNING: npcpy could not be installed automatically."
            echo "  Install it manually: python3 -m pip install npcpy"
        fi
    fi
fi

echo ""
echo "Run 'npcsh --version' or 'npc --version' to verify."
