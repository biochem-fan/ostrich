VERSION = 241011

def update_status(filename, message):
    if filename == "" or filename is None:
        return

    with open(filename, "w") as f:
        f.write("Status: " + message + "\n")
        f.flush()
