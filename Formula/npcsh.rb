class Npcsh < Formula
  desc "The composable multi-agent shell"
  homepage "https://github.com/NPC-Worldwide/npcsh"
  version "1.1.32"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/NPC-Worldwide/npcsh/releases/download/v#{version}/npcsh-macos-aarch64"
      sha256 "" # Updated by CI
    else
      url "https://github.com/NPC-Worldwide/npcsh/releases/download/v#{version}/npcsh-macos-x86_64"
      sha256 "" # Updated by CI
    end
  end

  on_linux do
    url "https://github.com/NPC-Worldwide/npcsh/releases/download/v#{version}/npcsh-linux-x86_64"
    sha256 "" # Updated by CI
  end

  def install
    binary = Dir["npcsh-*"].first || "npcsh"
    bin.install binary => "npcsh"
  end

  test do
    assert_match "npcsh", shell_output("#{bin}/npcsh --help 2>&1", 1)
  end
end
