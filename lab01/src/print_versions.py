"""Print and verify every package pinned by the lab manual."""
import platform
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version

root = Path(__file__).resolve().parents[2]
lines = [f'Python=={platform.python_version()}', f'Platform: {platform.platform()}', f'CPU: {platform.machine()}']
assert platform.python_version() == '3.11.8'
for requirement in (root / 'requirements.txt').read_text().splitlines():
    name, expected = requirement.split('==')
    actual = version(name)
    assert Version(actual) == Version(expected), (name, actual, expected)
    lines.append(f'{name}=={actual}')
text = '\n'.join(lines) + '\n'
(root / 'lab01/results/versions.txt').write_text(text)
print(text, end='')
