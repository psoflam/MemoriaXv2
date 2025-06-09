from memoriax2.config import DEBUG_MODE

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)  # Print only if DEBUG_MODE is True
    else:
        pass  # Suppress output if DEBUG_MODE is False

def silence_prints():
    import builtins
    builtins.print = lambda *args, **kwargs: None 