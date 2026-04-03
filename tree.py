import os
import argparse

import os
import argparse

def print_tree(start_path, prefix="", show_hidden=False, max_depth=None, max_files=None, current_depth=0):
    try:
        entries = sorted(os.listdir(start_path))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    if not show_hidden:
        entries = [e for e in entries if not e.startswith(".")]

    if max_files is not None:
        entries = entries[:max_files]

    entries_count = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if i == entries_count - 1 else "├── "

        print(prefix + connector + entry)

        # ✅ Only control recursion here (NOT at top of function)
        if os.path.isdir(path):
            if max_depth is None or current_depth + 1 < max_depth:
                extension = "    " if i == entries_count - 1 else "│   "
                print_tree(
                    path,
                    prefix + extension,
                    show_hidden,
                    max_depth,
                    max_files,
                    current_depth + 1
                )

def main():
    parser = argparse.ArgumentParser(description="Tree-like directory structure viewer")
    parser.add_argument("path", nargs="?", default=".", help="Directory path")
    parser.add_argument("-a", "--all", action="store_true", help="Show hidden files")
    parser.add_argument("-L", "--depth", type=int, help="Max display depth")
    parser.add_argument("--max-files", type=int, help="Max entries per directory")

    args = parser.parse_args()

    root = os.path.abspath(args.path)
    print(root)

    print_tree(
        root,
        show_hidden=args.all,
        max_depth=args.depth,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()