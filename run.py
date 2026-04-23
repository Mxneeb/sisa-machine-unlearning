"""Entry point — starts the SISA Unlearning Studio web app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.app import app

if __name__ == "__main__":
    print("=" * 55)
    print("  SISA Unlearning Studio")
    print("  Base paper: Bourtoule et al., IEEE S&P 2021")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 55)
    app.run(debug=False, port=5000)
