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

# Ensure ~/.npcsh/bin is on PATH.
case ":${PATH}:" in
    *":${INSTALL_DIR}:") ;;
    *)
        echo ""
        echo "Add ${INSTALL_DIR} to your PATH:"
        echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac

echo ""
echo "Run 'npcsh --version' or 'npc --version' to verify."
