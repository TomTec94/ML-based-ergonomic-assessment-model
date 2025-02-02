# main.py
import logging
from ui_tool import ErgoApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    logging.info("Launching the ErgoApp GUI...")
    app = ErgoApp()
    app.mainloop()

if __name__ == "__main__":
    main()