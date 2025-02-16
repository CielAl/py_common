import argparse

def int_float(value):
    """Convert to int if no decimal, otherwise convert to float."""
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid number: {value}")
