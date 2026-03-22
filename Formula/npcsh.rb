class Npcsh < Formula
  desc "The composable multi-agent shell"
  homepage "https://github.com/NPC-Worldwide/npcsh"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/NPC-Worldwide/npcsh/releases/latest/download/npcsh-macos-aarch64"
    else
      url "https://github.com/NPC-Worldwide/npcsh/releases/latest/download/npcsh-macos-x86_64"
    end
  end

  on_linux do
    url "https://github.com/NPC-Worldwide/npcsh/releases/latest/download/npcsh-linux-x86_64"
  end

  def install
    bin.install Dir["npcsh-*"].first || Dir["npcsh"].first => "npcsh"
  end

  test do
    system "true"
  end
end
