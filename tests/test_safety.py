from components.safety import catastrophic_reason

CATASTROPHIC = [
    ":(){ :|:& };:",
    "bomb(){ bomb|bomb& };bomb",
    "rm -rf /",
    "rm -rf /*",
    "rm -fr /",
    "rm -Rf /",
    "rm -r -f /",
    "rm --recursive --force /",
    "rm -rf /home",
    "rm -rf /home/",
    "rm -rf /home/*",
    "rm -rf ~",
    "rm -rf $HOME",
    "rm -rf \"$HOME\"",
    "rm -rf /root",
    "sudo rm -rf /",
    "echo cleaning up && rm -rf /home",
    "cd /tmp; rm -rf /",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/urandom of=/dev/nvme0n1 bs=1M",
    "mkfs.ext4 /dev/sda1",
    "wipefs -a /dev/sda",
    "yes | tr '\\n' 'x' > /dev/sda",
    "rm -rf --no-preserve-root /",
    "rm   -rf    /home",
    "FOO=bar rm -rf /home",
]

SAFE = [
    "rm -rf ./build",
    "rm -rf node_modules",
    "rm -rf /home/user/project/tmp",
    "rm -rf /tmp/scratch",
    "rm -rf /var/tmp/cache",
    "rm important_file.txt",
    "rm -f config.json",
    "find . -name '*.pyc' -delete",
    "git push --force origin main",
    "DROP TABLE sessions;",
    "dd if=/dev/sda of=backup.img",
    "ls -la /home",
    "mkdir -p /home/newuser",
    "echo 'rm -rf /' > README.md",
    "python -c \"print('rm -rf /home')\"",
    "rm -rf /home/user/.cache",
    "docker run --rm -v /home:/data busybox ls /data",
]


def test_catastrophic_commands_are_flagged():
    failures = [c for c in CATASTROPHIC if catastrophic_reason(c) is None]
    assert not failures, f"missed catastrophic commands: {failures}"


def test_safe_commands_are_not_flagged():
    failures = [c for c in SAFE if catastrophic_reason(c) is not None]
    details = [(c, catastrophic_reason(c)) for c in failures]
    assert not failures, f"false positives: {details}"


def test_reason_strings_are_informative():
    assert "fork bomb" in catastrophic_reason(":(){ :|:& };:")
    assert "rm -rf" in catastrophic_reason("rm -rf /home")
    assert "raw-disk" in catastrophic_reason("dd if=/dev/zero of=/dev/sda")


if __name__ == "__main__":
    test_catastrophic_commands_are_flagged()
    test_safe_commands_are_not_flagged()
    test_reason_strings_are_informative()
    print("all tests passed")
