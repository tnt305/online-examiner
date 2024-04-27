import ctypes

def check_remote_desktop():
    # Define necessary constants and types
    SM_REMOTESESSION = 0x1000
    ctypes.windll.user32.SystemParametersInfoW.restype = ctypes.c_bool
    
    # Check if the system is in a remote session
    remote_session = ctypes.windll.user32.SystemParametersInfoW(SM_REMOTESESSION, 0, None, 0)
    
    if remote_session:
        print("Warning: You are using Remote Desktop.")
        # You can also raise a warning instead of printing a message
    else:
        print("You are not using Remote Desktop.")

check_remote_desktop()
