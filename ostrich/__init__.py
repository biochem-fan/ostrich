VERSION = 250602
OSTRICH_ONLINE_SHM_NAME = "online_ostrich"

def update_status(filename, message):
    if filename == "" or filename is None:
        return

    with open(filename, "w") as f:
        f.write("Status: " + message + "\n")
        f.flush()
