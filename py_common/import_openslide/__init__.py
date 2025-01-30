import os
lib_path = os.path.join(os.path.dirname(__file__), 'bin')
if hasattr(os, "add_dll_directory"):
    # specify your own openslide binary locations
    with os.add_dll_directory(lib_path):
        import openslide
else:
    # os.environ['PATH'] = lib_path + ';' + os.environ['PATH']
    # #can either specify openslide bin path in PATH, or add it dynamically
    import openslide
