#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network Kivy Runner - Run the Maintenance Troubleshooting App on a different port
Fixed version with proper Kivy network configuration
"""

import os
import sys
import socket
import argparse
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Get the current directory to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Parent directory might be needed for imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a non-routable address to determine local IP
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def check_port_available(port):
    """Check if a port is available for use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('', port))
        sock.close()
        return True
    except OSError:
        return False


def setup_kivy_for_network(port):
    """Set up Kivy configuration for network access."""
    from kivy.config import Config

    # Configure Kivy before importing other Kivy modules
    Config.set('graphics', 'width', '1920')
    Config.set('graphics', 'height', '1080')
    Config.set('graphics', 'resizable', '1')

    # Network configuration
    Config.set('network', 'host', '0.0.0.0')  # Listen on all interfaces
    Config.set('network', 'port', str(port))

    # Input configuration for better remote access
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

    # Write the config
    Config.write()


def run_kivy_app(port):
    """Run the Kivy application with network configuration."""
    try:
        # Set up Kivy for network access
        setup_kivy_for_network(port)

        # Now import Kivy modules after configuration
        from kivy.core.window import Window

        # Set window properties for network access
        Window.bind(on_request_close=lambda *args: True)

        # Import and run the application
        from main_app import MaintenanceTroubleshootingApp

        print(f"Starting Kivy application on port {port}...")
        app = MaintenanceTroubleshootingApp()
        app.run()

    except ImportError as e:
        print(f"Error importing MaintenanceTroubleshootingApp: {e}")
        print("Make sure this script is in the same directory as main_app.py")
        return False
    except Exception as e:
        print(f"Error running the Kivy application: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


class KivyHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving Kivy app remotely."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            # Serve a simple HTML page that connects to the Kivy app
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Maintenance Troubleshooting App</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        margin: 50px;
                        background-color: #f0f0f0;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        max-width: 600px;
                        margin: 0 auto;
                    }}
                    .button {{
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 15px 32px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 4px;
                    }}
                    .info {{
                        background-color: #e7f3ff;
                        border-left: 6px solid #2196F3;
                        padding: 20px;
                        margin: 20px 0;
                        text-align: left;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Maintenance Troubleshooting Application</h1>
                    <p>The Kivy application is running on this server.</p>

                    <div class="info">
                        <h3>Access Information:</h3>
                        <p><strong>Server:</strong> {get_local_ip()}:{self.server.server_port}</p>
                        <p><strong>Status:</strong> Application Running</p>
                        <p><strong>Type:</strong> Kivy-based Desktop Application</p>
                    </div>

                    <h3>How to Connect:</h3>
                    <ol style="text-align: left;">
                        <li>The Kivy application should be running in a window on the server machine</li>
                        <li>For remote access, you may need to use VNC or RDP to connect to the server desktop</li>
                        <li>Alternatively, use Kivy's built-in remote debugging features</li>
                    </ol>

                    <p style="margin-top: 30px;">
                        <em>Note: This is a desktop application. For full remote access, 
                        consider using remote desktop software.</em>
                    </p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()


def start_web_server(port):
    """Start a simple web server to provide information about the Kivy app."""
    web_port = port + 1000  # Use a different port for the web interface

    try:
        server = HTTPServer(('0.0.0.0', web_port), KivyHTTPHandler)
        print(f"Web interface available at: http://{get_local_ip()}:{web_port}")
        server.serve_forever()
    except Exception as e:
        print(f"Could not start web server: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run Kivy app on a different port')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port for the application (default: 8000)')
    parser.add_argument('--no-web', action='store_true',
                        help='Disable the web interface')
    args = parser.parse_args()

    local_ip = get_local_ip()

    # Check if the port is available
    if not check_port_available(args.port):
        print(f"Port {args.port} is already in use. Please choose a different port.")
        sys.exit(1)

    print(f"\n=== Starting Maintenance Troubleshooting Application ===")
    print(f"Local IP Address: {local_ip}")
    print(f"Application Port: {args.port}")
    print(f"Network Access: http://{local_ip}:{args.port}")

    if not args.no_web:
        web_port = args.port + 1000
        if check_port_available(web_port):
            print(f"Web Interface: http://{local_ip}:{web_port}")
            # Start web server in a separate thread
            web_thread = threading.Thread(target=start_web_server, args=(args.port,))
            web_thread.daemon = True
            web_thread.start()
        else:
            print(f"Web interface port {web_port} is in use, skipping web interface")

    print("Press Ctrl+C to stop the application")
    print("=" * 55)
    print()

    # Run the Kivy application
    try:
        success = run_kivy_app(args.port)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()