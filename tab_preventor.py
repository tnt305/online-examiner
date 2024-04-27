from flask import Flask, render_template, request
import threading
import tkinter as tk
import tkinter.messagebox as messagebox
import keyboard

app = Flask(__name__)

def tab_tracker():
    tab_count = 1  # Initialize tab count

    def on_alt_tab(event):
        nonlocal tab_count
        if keyboard.is_pressed('alt') and event.name == 'tab':
            tab_count += 1
            message = f'Tab switched. Number of tabs: {tab_count-1}'
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
    keyboard.on_press(on_alt_tab)

    # Block until KeyboardInterrupt (Ctrl+C) is detected
    try:
        keyboard.wait('esc')  # Wait for the 'esc' key to exit
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        keyboard.unhook_all()

def start_tab_tracker():
    thread = threading.Thread(target=tab_tracker)
    thread.start()

@app.route('/')
def index():
    start_tab_tracker()  # Start tab tracker in a separate thread
    return render_template('calculator.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    num1 = float(request.form['num1'])
    num2 = float(request.form['num2'])
    operation = request.form['operation']

    result = 0

    if operation == 'add':
        result = num1 + num2
    elif operation == 'subtract':
        result = num1 - num2
    elif operation == 'multiply':
        result = num1 * num2
    elif operation == 'divide':
        if num2 != 0:
            result = num1 / num2
        else:
            return "Error: Cannot divide by zero"

    return render_template('result.html', num1=num1, num2=num2, operation=operation, result=result)

if __name__ == '__main__':
    app.run(debug=True)
