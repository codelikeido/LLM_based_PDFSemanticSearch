"""
main.py - Entry point for the PDF Semantic Search backend.

Run:
  python main.py                 # development server
  gunicorn main:app              # production (gunicorn must be installed)
  gunicorn -w 4 -b 0.0.0.0:5000 main:app
"""
import os
from app.factory import create_app

app = create_app()

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    host  = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "true").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════╗
║        PDF Semantic Search Backend  v1.0.0           ║
╠══════════════════════════════════════════════════════╣
║  Host  : {host:<44}║
║  Port  : {str(port):<44}║
║  Debug : {str(debug):<44}║
╠══════════════════════════════════════════════════════╣
║  Endpoints:                                          ║
║   POST  /api/v1/documents/upload                     ║
║   GET   /api/v1/documents                            ║
║   GET   /api/v1/documents/<doc_id>                   ║
║   DELETE /api/v1/documents/<doc_id>                  ║
║   POST  /api/v1/search                               ║
║   GET   /api/v1/health                               ║
╚══════════════════════════════════════════════════════╝
    """)

    app.run(host=host, port=port, debug=debug)
