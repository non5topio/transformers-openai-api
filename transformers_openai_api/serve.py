from flask import Flask
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_server(app: Flask):
    try:
        host = app.config.get('HOST', '127.0.0.1')
        port = app.config.get('PORT', 8001)
        debug = app.config.get('ENV', 'production') != 'production'
        
        logger.info(f"Starting server on {host}:{port} (debug={debug})")
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,  # Enable threading for better concurrent handling
            use_reloader=False  # Disable reloader in production
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
