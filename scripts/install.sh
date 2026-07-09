#!/bin/sh
# Install the latest npcsh Rust binary.
# Usage: curl -fsSL https://enpisi.com/install-npcsh.sh | sh

set -e

REPO="NPC-Worldwide/npcsh"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.npcsh/bin}"
BIN_NAME="npcrsh"
SHELL_NAME="npcsh"

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

if [ "$OS" = "windows" ]; then
    echo "Windows install is not supported by this script yet. Use cargo install npcsh." >&2
    exit 1
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

ASSET="npcrsh-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

echo "Installing npcsh ${TAG} for ${OS}/${ARCH}..."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

TMP_BIN="${TMP_DIR}/npcrsh"
curl -fsSL "$URL" -o "$TMP_BIN"
chmod +x "$TMP_BIN"

mkdir -p "$INSTALL_DIR"
INSTALL_PATH="${INSTALL_DIR}/${BIN_NAME}"
cp "$TMP_BIN" "$INSTALL_PATH"

echo "Binary installed to ${INSTALL_PATH}"

# Optionally symlink npcsh -> npcrsh for the shell entry point.
LINK_PATH="${INSTALL_DIR}/${SHELL_NAME}"
if [ ! -e "$LINK_PATH" ] && [ "$(basename "$LINK_PATH")" != "$(basename "$INSTALL_PATH")" ]; then
    ln -sf "$INSTALL_PATH" "$LINK_PATH"
    echo "Linked ${SHELL_NAME} -> ${BIN_NAME}"
fi

# macOS: if the downloaded binary was not signed by us, ad-hoc sign it so
# Gatekeeper does not kill it on first run. This is a fallback only.
if [ "$OS" = "macos" ] && command -v codesign >/dev/null 2>&1; then
    if ! codesign -v "$INSTALL_PATH" >/dev/null 2>&1; then
        echo "Applying ad-hoc signature for macOS Gatekeeper..."
        codesign -s - -f "$INSTALL_PATH" >/dev/null
    fi
fi

# Ensure ~/.npcsh/bin is on PATH.
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo ""
        echo "Add ${INSTALL_DIR} to your PATH:"
        echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac

echo ""
echo "Run 'npcsh --version' to verify."
