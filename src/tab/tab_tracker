import tkinter as tk
from tkinter import messagebox
import keyboard


def tab_tracker():
    tab_count = 1  # Initialize tab count

    def on_tab_switch(event):
        nonlocal tab_count
        if (keyboard.is_pressed('alt') and event.name == 'tab') or \
           (keyboard.is_pressed('ctrl') and event.name == 'tab') or \
           (keyboard.is_pressed('win') and event.name == 'tab'):
            tab_count += 1
            message = f'Tab switched detected with total of: {tab_count-1} times' 
            print(message)
            show_notification(message)

    def show_notification(message):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Create a message box
        messagebox.showinfo("Tab Switch Notification", message)

        # Destroy the Tkinter root window after showing the message
        root.destroy()

    # Register the tab switch event listener
    keyboard.on_press(on_tab_switch)

    # Block until KeyboardInterrupt (Ctrl+C) is detected
    try:
        keyboard.wait('esc')  # Wait for the 'esc' key to exit
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        keyboard.unhook_all()