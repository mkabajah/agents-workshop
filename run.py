"""
One entry point for every framework. LangChain/LangGraph is the default.

    python run.py 02 "Which candidates know Kafka?"                    # LangChain (session1/02_first_agent.py)
    python run.py 02 "Which candidates know Kafka?" --framework strands
    python run.py 04 --framework crewai --auto
    python run.py --list                                               # what exists for each framework

Steps: 02 first agent · 03 memory · 04 limits + human approval · 05 your own guardrail · 06 workflow
(01, the loop by hand, has no framework: python session1/01_loop_by_hand.py)
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
FOLDERS = {"langchain": ROOT / "session1", "strands": ROOT / "frameworks" / "strands", "crewai": ROOT / "frameworks" / "crewai"}


def find(framework: str, step: str) -> pathlib.Path:
    """The script for one step of one framework.
    e.g. ("strands", "2") → frameworks/strands/02_first_agent.py"""
    hits = sorted(FOLDERS[framework].glob(f"{step.zfill(2)}_*.py"))
    if not hits:
        sys.exit(f"No step {step} for {framework}. Try: python run.py --list")
    return hits[0]


def main(argv):
    """Read the arguments, pick the script, and run it with the remaining arguments.
    Returns the script's exit code, so tests can check that it worked."""
    if "--list" in argv or not argv:
        for name, folder in FOLDERS.items():
            print(f"{name:10}", " · ".join(p.stem for p in sorted(folder.glob("0[2-6]_*.py"))))
        return 0
    framework = "langchain"
    if "--framework" in argv:
        i = argv.index("--framework")
        framework = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if framework not in FOLDERS:
        sys.exit(f"--framework must be one of {', '.join(FOLDERS)}")
    script = find(framework, argv[0])
    print(f"▶ {framework}: {script.relative_to(ROOT)}\n")
    # Run the script as its own Python process (same Python, same arguments), exactly as if you had
    # typed `python <script> …` yourself.
    return subprocess.call([sys.executable, str(script), *argv[1:]])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
