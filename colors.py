# ANSI color codes for terminal output


class Colors:
    """ANSI color codes for colorful terminal output"""
    GREEN = '\033[92m'     # Green for success/100% matches
    YELLOW = '\033[93m'    # Yellow for warnings/90-99% matches
    CYAN = '\033[96m'      # Cyan for info
    RED = '\033[91m'       # Red for errors
    BLUE = '\033[94m'      # Blue for info
    MAGENTA = '\033[95m'   # Magenta for highlights
    WHITE = '\033[97m'     # White for emphasis
    RESET = '\033[0m'      # Reset to normal color


# Alternative function-based interface for those who prefer it
def green(text):
    """Return text colored in green"""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def yellow(text):
    """Return text colored in yellow"""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def cyan(text):
    """Return text colored in cyan"""
    return f"{Colors.CYAN}{text}{Colors.RESET}"


def red(text):
    """Return text colored in red"""
    return f"{Colors.RED}{text}{Colors.RESET}"


def blue(text):
    """Return text colored in blue"""
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def magenta(text):
    """Return text colored in magenta"""
    return f"{Colors.MAGENTA}{text}{Colors.RESET}"


def white(text):
    """Return text colored in white"""
    return f"{Colors.WHITE}{text}{Colors.RESET}"


# Success/Error helpers
def success(text):
    """Return text colored as success (green)"""
    return green(text)


def warning(text):
    """Return text colored as warning (yellow)"""
    return yellow(text)


def error(text):
    """Return text colored as error (red)"""
    return red(text)


def info(text):
    """Return text colored as info (cyan)"""
    return cyan(text)
