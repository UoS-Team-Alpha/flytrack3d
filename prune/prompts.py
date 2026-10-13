"""Questions asked in the terminal. Every prompt takes q to quit."""

from __future__ import annotations


def prompt_choice(options, prompt_text):
    """Ask the user to pick an option by number until a valid one is given.
    Returns None if the user types 'q' to quit."""
    while True:
        choice = input(prompt_text).strip()
        if choice.lower() == 'q':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Invalid choice. Enter a number between 1 and {len(options)}, or q to quit.")


def prompt_yes_no(prompt_text):
    """Ask until the user types y or n. Returns True/False, or None if they type 'q' to quit."""
    while True:
        choice = input(prompt_text).strip().lower()
        if choice in ('y', 'n'):
            return choice == 'y'
        if choice == 'q':
            return None
        print("Invalid choice. Enter y, n, or q to quit.")
