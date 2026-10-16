from core.logging_setup import init_logging
from ui.main_dashboard import MeshUnwarpApp

if __name__ == "__main__":
    log = init_logging(role="main")
    log.info("MeshUnwarp starting")
    app = MeshUnwarpApp()
    app.mainloop()
