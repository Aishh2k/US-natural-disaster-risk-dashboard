import http.server
import socketserver
import os
import webbrowser
import json
import sys

# Add preprocessing directory to path to import chatbot_api
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessing"))
import chatbot_api

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                message = data.get('message')
                context = data.get('context')
                
                response_data = chatbot_api.get_openai_response(message, context)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_error(404, "File not found")

# Check for API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("\n" + "="*60)
    print("WARNING: GROQ_API_KEY environment variable not found!")
    print("The chatbot feature will not work without it.")
    print("Please set it using:")
    print("  PowerShell: $env:GROQ_API_KEY = 'your-key'")
    print("  CMD: set GROQ_API_KEY=your-key")
    print("Then RESTART this server.")
    print("="*60 + "\n")
else:
    print(f"\nSUCCESS: GROQ_API_KEY found (starts with {api_key[:5]}...)\n")

print(f"Serving dashboard at http://localhost:{PORT}")
print("Press Ctrl+C to stop.")

# Open browser automatically
webbrowser.open(f"http://localhost:{PORT}")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
