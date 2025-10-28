#!/usr/bin/env python3
"""
AI-Enhanced EMTAC Application
Enhanced with model preloading and offline optimization
"""

# ========================================
# OFFLINE MODE CONFIGURATION - Must be FIRST!
# ========================================
import os

print("CONFIGURING OFFLINE MODE - network checks disabled for AI models")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================================
# UNICODE CONFIGURATION - Must be VERY EARLY!
# ========================================
import sys


def configure_unicode_environment():
    """Configure environment for proper Unicode handling."""

    # Set Python encoding environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'

    # Configure sys.stdout and stderr for Unicode
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    print("Unicode environment configured successfully")


# CALL THIS IMMEDIATELY - BEFORE ANY OTHER MAJOR IMPORTS
configure_unicode_environment()

# ========================================
# NOW CONTINUE WITH REGULAR IMPORTS
# ========================================
# ai_emtac.py - Using custom request ID logging system
from datetime import datetime
import webbrowser
import socket
import time
from threading import Timer, Thread
from flask import Flask, session, request, redirect, url_for, current_app, render_template, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import logging system first
from modules.configuration.log_config import (
    initial_log_cleanup, logger,
    debug_id, info_id, warning_id, error_id, critical_id,
    get_request_id, set_request_id, with_request_id, log_timed_operation,
    request_id_middleware
)

# ========== CRITICAL: DATABASE SERVICE CHECK BEFORE DATABASE IMPORTS ==========
# Import database service check function
from modules.database_manager.db_manager import ensure_database_service_running, create_document_structure_tables

# Set up request ID for this critical check
early_startup_request_id = set_request_id("pre-import-db-check")

info_id("[PRE-IMPORT] Checking PostgreSQL database service availability...", early_startup_request_id)
print("[PRE-IMPORT] Checking PostgreSQL database service availability...")

with log_timed_operation("pre_import_database_service_check", early_startup_request_id):
    if not ensure_database_service_running():
        error_id("PostgreSQL database service is not available before imports", early_startup_request_id)
        print("PostgreSQL database service is not available")
        print("Please check:")
        print("   - PostgreSQL is installed and running")
        print("   - .env file contains correct database credentials")
        print("   - Windows Services has PostgreSQL service started")
        critical_id("Application startup aborted due to database unavailability", early_startup_request_id)
        sys.exit(1)

info_id("[PRE-IMPORT] PostgreSQL database service is ready - proceeding with imports", early_startup_request_id)
print("[PRE-IMPORT] PostgreSQL database service is ready - proceeding with imports")
# ========== END CRITICAL DATABASE SERVICE CHECK ==========

# NOW we can safely import modules that connect to the database
from modules.emtacdb.emtacdb_fts import UserLogin, initialize_database_tables
from utilities.custom_jinja_filters import register_jinja_filters
from modules.emtacdb.emtacdb_fts import (UserLevel)
from modules.emtacdb.utlity.main_database.database import serve_image, db_config
from blueprints import register_blueprints
from modules.emtacdb.utlity.revision_database.event_listeners import register_event_listeners
from modules.configuration.config import UPLOAD_FOLDER, DATABASE_URL
from utilities.auth_utils import requires_roles
from modules.configuration.config_env import DatabaseConfig



# ========================================
# ENHANCED DATABASE ENGINE WITH UNICODE SUPPORT
# ========================================
def create_unicode_engine(database_url):
    """Create SQLAlchemy engine with proper Unicode support."""

    # Add encoding parameters if not already present
    if '?' in database_url:
        enhanced_url = database_url + "&client_encoding=utf8&connect_timeout=30"
    else:
        enhanced_url = database_url + "?client_encoding=utf8&connect_timeout=30"

    engine = create_engine(
        enhanced_url,
        # Ensure connection uses UTF-8
        connect_args={
            'client_encoding': 'utf8',
        },
        # Connection pool settings
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Set to True for debugging SQL
    )

    return engine

db_config= DatabaseConfig()
# Initialize main database engine (PostgreSQL) with Unicode support
engine = create_unicode_engine(DATABASE_URL)
# Add this to your database setup
create_document_structure_tables(db_config)

# Create main session factory (PostgreSQL only)
SessionFactory = sessionmaker(bind=engine)

# Global variables for model preloading status
_model_preload_status = {
    'started': False,
    'completed': False,
    'error': None,
    'start_time': None,
    'completion_time': None,
    'request_id': None
}


@with_request_id
def configure_offline_mode(request_id=None):
    """Configure environment for offline model loading - MAJOR SPEEDUP!"""
    offline_env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }

    for key, value in offline_env_vars.items():
        os.environ[key] = value

    info_id("Configured offline mode - network checks disabled for AI models", request_id)
    print("Configured offline mode - network checks disabled for AI models")


def preload_ai_models():
    """Preload AI models in background for instant access - ENHANCED with embedding support"""
    global _model_preload_status

    # Set request ID for this background operation
    request_id = set_request_id("model-preload")
    _model_preload_status['request_id'] = request_id

    try:
        with log_timed_operation("ai_model_preloading", request_id):
            _model_preload_status['started'] = True
            _model_preload_status['start_time'] = time.time()

            info_id("Starting AI model preloading (IMAGE + EMBEDDING)...", request_id)
            print("Starting AI model preloading (IMAGE + EMBEDDING)...")

            # Import and preload models
            from plugins.ai_modules.ai_models import ModelsConfig

            # =========================
            # PRELOAD IMAGE MODEL (existing)
            # =========================
            try:
                model_handler = ModelsConfig.load_image_model()
                model_type = type(model_handler).__name__

                info_id(f"Preloaded image model: {model_type}", request_id)
                print(f"Preloaded image model: {model_type}")

                # If it's CLIPModelHandler, get cache stats
                if hasattr(model_handler, 'get_cache_stats'):
                    cache_stats = model_handler.get_cache_stats()
                    debug_id(f"Image model cache stats: {cache_stats}", request_id)

            except Exception as model_error:
                warning_id(f"Could not preload image model: {model_error}", request_id)
                print(f"Could not preload image model: {model_error}")

            # =========================
            # PRELOAD EMBEDDING MODEL (NEW!)
            # =========================
            try:
                info_id("Loading embedding model for vector search...", request_id)
                print("Loading embedding model for vector search...")

                # Get current embedding model configuration
                current_ai_model, current_embedding_model = ModelsConfig.load_config_from_db()

                if current_embedding_model and current_embedding_model != 'NoEmbeddingModel':
                    info_id(f"Preloading embedding model: {current_embedding_model}", request_id)
                    print(f"Preloading embedding model: {current_embedding_model}")

                    # Force load the embedding model
                    embedding_handler = ModelsConfig.load_embedding_model(current_embedding_model)

                    # Test that it works by generating a test embedding
                    test_embedding = embedding_handler.get_embeddings("startup test embedding")

                    if test_embedding and len(test_embedding) > 0:
                        info_id(f"Embedding model preloaded successfully (dim: {len(test_embedding)})", request_id)
                        print(f"Embedding model preloaded successfully (dim: {len(test_embedding)})")

                        # Store model readiness flag
                        _model_preload_status['embedding_ready'] = True
                        _model_preload_status['embedding_dimension'] = len(test_embedding)
                    else:
                        warning_id("Embedding model loaded but test embedding failed", request_id)
                        print("Embedding model loaded but test embedding failed")
                        _model_preload_status['embedding_ready'] = False
                else:
                    info_id("No embedding model configured - skipping embedding preload", request_id)
                    print("No embedding model configured - skipping embedding preload")
                    _model_preload_status['embedding_ready'] = False

            except Exception as embedding_error:
                error_id(f"Failed to preload embedding model: {embedding_error}", request_id)
                print(f"Failed to preload embedding model: {embedding_error}")
                _model_preload_status['embedding_ready'] = False
                _model_preload_status['embedding_error'] = str(embedding_error)

            _model_preload_status['completed'] = True
            _model_preload_status['completion_time'] = time.time()

            preload_time = _model_preload_status['completion_time'] - _model_preload_status['start_time']
            info_id(f"AI model preloading completed in {preload_time:.2f}s", request_id)
            print(f"AI model preloading completed in {preload_time:.2f}s")

    except Exception as e:
        _model_preload_status['error'] = str(e)
        _model_preload_status['completed'] = True  # Mark as completed even with error

        error_id(f"Error during model preloading: {e}", request_id, exc_info=True)
        print(f"Error during model preloading: {e}")


# Enhanced model preload status structure
_model_preload_status = {
    'started': False,
    'completed': False,
    'error': None,
    'start_time': None,
    'completion_time': None,
    'request_id': None,
    # NEW: Embedding model specific status
    'embedding_ready': False,
    'embedding_dimension': None,
    'embedding_error': None
}


def check_embedding_model_readiness():
    """Check if embedding model is preloaded and ready for vector search"""
    global _model_preload_status

    if not _model_preload_status['completed']:
        return {
            'ready': False,
            'reason': 'Model preloading not completed yet',
            'status': 'loading'
        }

    if _model_preload_status.get('embedding_ready', False):
        return {
            'ready': True,
            'reason': 'Embedding model preloaded successfully',
            'dimension': _model_preload_status.get('embedding_dimension'),
            'status': 'ready'
        }

    return {
        'ready': False,
        'reason': _model_preload_status.get('embedding_error', 'Unknown embedding model issue'),
        'status': 'error'
    }

@with_request_id
def preload_models_async(request_id=None):
    """Start model preloading in background thread"""
    thread = Thread(target=preload_ai_models, daemon=True, name="ModelPreloader")
    thread.start()
    info_id("Started model preloading thread", request_id)
    return thread


def get_local_ip():
    """
    Dynamically retrieves the local IP address by creating a temporary socket
    to a public DNS server (8.8.8.8).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip


def open_browser():
    """Open browser with request ID tracking"""
    request_id = set_request_id("browser-open")
    port = int(os.environ.get('PORT', 5000))
    ip = get_local_ip()
    url = f'http://{ip}:{port}/'
    info_id(f"Opening browser at {url}", request_id)
    webbrowser.open_new(url)


@with_request_id
def create_app(request_id=None):
    """Create Flask application with optimized model loading and request ID tracking"""

    with log_timed_operation("flask_app_creation", request_id):
        app = Flask(__name__)

        # Set the secret key for session encryption
        app.secret_key = '1234'  # Replace with a secure secret key for production

        # Configure Flask for Unicode support
        app.config['JSON_AS_ASCII'] = False  # Allow Unicode in JSON responses

        # Register custom Jinja filters
        register_jinja_filters(app)

        # Set the upload folder in the app's configuration
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        # Initialize and set db_config
        db_config = DatabaseConfig()
        app.config['db_config'] = db_config

        # ========== REQUEST ID MIDDLEWARE ==========

        # Apply custom request ID middleware for automatic tracking
        request_id_middleware(app)
        info_id("Request ID middleware registered", request_id)

        # ========== MODEL PRELOADING OPTIMIZATION ==========

        # Configure offline mode for faster model loading
        configure_offline_mode(request_id)

        # Initialize models configuration
        from plugins.ai_modules import ModelsConfig
        ModelsConfig.initialize_models_config_table()
        info_id("Models configuration initialized during app startup", request_id)
        print("Models configuration initialized during app startup")

        # Start AI model preloading in background
        info_id("Starting AI model preloading for faster image processing...", request_id)
        print("Starting AI model preloading for faster image processing...")
        preload_thread = preload_models_async(request_id)

        # ========== END MODEL PRELOADING SECTION ==========



        # Register blueprints and event listeners
        register_blueprints(app)
        register_event_listeners()
        app.has_cleared_session = True

        @app.before_request
        def global_login_check():
            endpoint = request.endpoint
            request_id = get_request_id()  # Get current request ID
            debug_id(f"Incoming endpoint: {endpoint}", request_id)

            # Only proceed with the activity tracking if user is logged in and has a login record
            if 'user_id' in session and 'login_record_id' in session:
                try:
                    with log_timed_operation("user_activity_update", request_id):
                        # Use PostgreSQL session for user activity tracking
                        with SessionFactory() as session_db:
                            login_record = session_db.query(UserLogin).get(session['login_record_id'])
                            if login_record and login_record.is_active:
                                login_record.last_activity = datetime.utcnow()
                                session_db.commit()
                                debug_id(f"Updated activity for user {session['user_id']}", request_id)
                except Exception as e:
                    error_id(f"Error updating activity timestamp: {e}", request_id, exc_info=True)

            # Continue with login check
            allowed_routes = [
                'login_bp.login',
                'login_bp.logout',
                'static',  # Allow static files
                'create_user_bp.create_user',
                'create_user_bp.submit_user_creation',
                'health',  # Allow health checks
                'model_status',  # Allow model status checks
                'api_status'  # Allow API status checks
            ]

            if request.endpoint is None:
                return

            if 'user_id' not in session and request.endpoint not in allowed_routes:
                debug_id(f"Redirecting unauthenticated user from {endpoint} to login", request_id)
                return redirect(url_for('login_bp.login'))

            session.permanent = True

        @app.route('/')
        def index():
            request_id = get_request_id()
            session.permanent = False
            user_id = session.get('user_id', '')
            user_level = session.get('user_level', UserLevel.STANDARD.value)

            debug_id(f"Index page accessed by user {user_id} with level {user_level}", request_id)

            if not user_id:
                debug_id("No user_id in session, redirecting to login", request_id)
                return redirect(url_for('login_bp.login'))

            try:
                with log_timed_operation("load_model_config", request_id):
                    # Load current AI model configuration
                    from plugins.ai_modules import ModelsConfig
                    current_ai_model, current_embedding_model = ModelsConfig.load_config_from_db()
                    debug_id(f"Loaded models: AI={current_ai_model}, Embedding={current_embedding_model}", request_id)
            except Exception as e:
                error_id(f"Error loading model configuration: {e}", request_id, exc_info=True)
                current_ai_model, current_embedding_model = "Error", "Error"

            return render_template('index.html',
                                   current_ai_model=current_ai_model,
                                   current_embedding_model=current_embedding_model,
                                   user_level=user_level)

        # ========== MODEL STATUS AND HEALTH CHECK ROUTES ==========

        @app.route('/health')
        def health_check():
            """Health check endpoint with model preloading status"""
            global _model_preload_status
            request_id = get_request_id()

            debug_id("Health check requested", request_id)

            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "postgresql_connected",
                "unicode_support": "enabled",
                "request_id": request_id,
                "model_preloading": {
                    "started": _model_preload_status['started'],
                    "completed": _model_preload_status['completed'],
                    "error": _model_preload_status['error'],
                    "duration": None,
                    "preload_request_id": _model_preload_status.get('request_id')
                }
            }

            # Calculate preloading duration if available
            if _model_preload_status['start_time'] and _model_preload_status['completion_time']:
                duration = _model_preload_status['completion_time'] - _model_preload_status['start_time']
                health_data["model_preloading"]["duration"] = f"{duration:.2f}s"

            # Get model cache stats if available
            try:
                from plugins.ai_modules.ai_models import ModelsConfig
                model_handler = ModelsConfig.load_image_model()
                if hasattr(model_handler, 'get_cache_stats'):
                    health_data["model_cache"] = model_handler.get_cache_stats()
                    debug_id(f"Model cache stats retrieved: {health_data['model_cache']}", request_id)
            except Exception as e:
                health_data["model_cache"] = {"error": str(e)}
                warning_id(f"Could not get model cache stats: {e}", request_id)

            return jsonify(health_data)

        @app.route('/model-status')
        def model_status():
            """Detailed model status endpoint"""
            global _model_preload_status
            request_id = get_request_id()

            debug_id("Model status requested", request_id)

            try:
                with log_timed_operation("model_status_collection", request_id):
                    from plugins.ai_modules.ai_models import ModelsConfig

                    status_data = {
                        "preload_status": _model_preload_status.copy(),
                        "current_models": {},
                        "cache_stats": {},
                        "performance_ready": False,
                        "database_type": "postgresql",
                        "unicode_support": "enabled",
                        "request_id": request_id
                    }

                    # Get current model configuration
                    try:
                        current_ai_model, current_embedding_model = ModelsConfig.load_config_from_db()
                        status_data["current_models"] = {
                            "image_model": current_ai_model,
                            "embedding_model": current_embedding_model
                        }
                        debug_id(f"Current models retrieved: {status_data['current_models']}", request_id)
                    except Exception as e:
                        status_data["current_models"]["error"] = str(e)
                        warning_id(f"Error getting current models: {e}", request_id)

                    # Get cache statistics
                    try:
                        model_handler = ModelsConfig.load_image_model()
                        if hasattr(model_handler, 'get_cache_stats'):
                            cache_stats = model_handler.get_cache_stats()
                            status_data["cache_stats"] = cache_stats
                            status_data["performance_ready"] = cache_stats.get("models_cached", 0) > 0
                            debug_id(f"Performance ready: {status_data['performance_ready']}", request_id)
                    except Exception as e:
                        status_data["cache_stats"]["error"] = str(e)
                        warning_id(f"Error getting cache stats: {e}", request_id)

                    return jsonify(status_data)

            except Exception as e:
                error_id(f"Error in model_status endpoint: {e}", request_id, exc_info=True)
                return jsonify({
                    "error": str(e),
                    "preload_status": _model_preload_status.copy(),
                    "request_id": request_id
                }), 500

        @app.route('/api-status')
        def api_status():
            """Quick API status check"""
            request_id = get_request_id()
            debug_id("API status requested", request_id)

            return jsonify({
                "api": "ready",
                "database": "postgresql",
                "unicode_support": "enabled",
                "models_preloaded": _model_preload_status['completed'],
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            })

        # ========== END NEW ROUTES ==========

        @app.route('/upload_search_database')
        def upload_image_page():
            request_id = get_request_id()
            session.permanent = False
            debug_id("Upload search database page accessed", request_id)
            total_pages = 1
            page = 1
            return render_template('upload_search_database/upload_search_database.html', total_pages=total_pages,
                                   page=page)

        @app.route('/success')
        def upload_success():
            request_id = get_request_id()
            session.permanent = False
            debug_id("Upload success page accessed", request_id)
            return render_template('success.html')

        @app.route('/view_pdf_by_title/<string:title>')
        def view_pdf_by_title_route(title):
            request_id = get_request_id()
            session.permanent = False
            debug_id(f"PDF view requested for title: {title}", request_id)
            return view_pdf_by_title(title)

        @app.route('/serve_image/<int:image_id>')
        def serve_image_route(image_id):
            request_id = get_request_id()
            debug_id(f"Request to serve image with ID: {image_id}", request_id)

            with SessionFactory() as session:
                try:
                    with log_timed_operation(f"serve_image_{image_id}", request_id):
                        return serve_image(session, image_id)
                except Exception as e:
                    error_id(f"Error serving image {image_id}: {e}", request_id, exc_info=True)
                    return "Image not found", 404

        @app.route('/document_success')
        def document_upload_success():
            request_id = get_request_id()
            session.permanent = False
            debug_id("Document upload success page accessed", request_id)
            return render_template('success.html')

        @app.route('/troubleshooting_guide')
        @requires_roles(UserLevel.ADMIN.value, UserLevel.LEVEL_III.value)
        def troubleshooting_guide():
            request_id = get_request_id()
            session.permanent = False
            debug_id("Troubleshooting guide accessed", request_id)
            return render_template('troubleshooting_guide.html')

        @app.route('/tsg_search_problems')
        def tsg_search_problems():
            request_id = get_request_id()
            session.permanent = False
            debug_id("TSG search problems page accessed", request_id)
            return render_template('tsg_search_problems.html')

        @app.route('/search_bill_of_material', methods=['GET'])
        def search_bill_of_material():
            request_id = get_request_id()
            debug_id("Search bill of material page accessed", request_id)
            return render_template('search_bill_of_material.html')

        @app.route('/bill_of_materials')
        def bill_of_materials():
            request_id = get_request_id()
            debug_id("Bill of materials page accessed", request_id)
            return render_template('bill_of_materials/bill_of_materials.html')

        @app.route('/position_data_assignment')
        def position_data_assignment():
            request_id = get_request_id()
            debug_id("Position data assignment page accessed", request_id)
            return render_template('position_data_assignment/position_data_assignment.html')

        @app.errorhandler(403)
        def forbidden(e):
            request_id = get_request_id()
            warning_id(f"403 Forbidden error: {e}", request_id)
            return render_template('403.html'), 403

        # Print URL rules for debugging (optional - can be removed in production)
        if app.debug:
            for rule in app.url_map.iter_rules():
                print(rule)

        # Log configuration details after the app is created
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        ip = get_local_ip()
        url = f'http://{ip}:{port}/'
        info_id(f"Starting application on host: {host}, port: {port}", request_id)
        info_id(f"Accessible at: {url}", request_id)
        print(f"Starting application on host: {host}, port: {port}")
        print(f"Accessible at: {url}")

        return app


if __name__ == '__main__':
    """Must run in terminal python ai_emtac.py to allow remote access to local network"""

    # Set up request ID for startup process
    startup_request_id = set_request_id("app-startup")

    print('Perform initial log cleanup (compress old logs and delete old backups)')
    with log_timed_operation("initial_log_cleanup", startup_request_id):
        initial_log_cleanup()

    # Database service was already checked before imports, so we can skip duplicate check here
    info_id("Database service already verified during import phase", startup_request_id)
    info_id("Unicode environment configured for proper character handling", startup_request_id)
    print("Database service already verified during import phase")
    print("Unicode environment configured for proper character handling")

    # Read configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'  # Default to debug mode if not set

    ip = get_local_ip()
    url = f'http://{ip}:{port}/'

    # Log the configuration details when running the script directly
    info_id(f"Starting application on host: {host}, port: {port}", startup_request_id)
    info_id(f"Accessible at: {url}", startup_request_id)
    print(f"Starting application on host: {host}, port: {port}")
    print(f"Accessible at: {url}")

    # Optional: Open the browser after a slight delay (3 seconds to allow model preloading)
    Timer(3, open_browser).start()

    # Create and run the application (database is now guaranteed to be ready)
    with log_timed_operation("create_flask_app", startup_request_id):
        app = create_app(startup_request_id)

    # Show helpful URLs
    print("\n" + "=" * 60)
    print("EMTAC AI APPLICATION READY")
    print("=" * 60)
    print(f"Main Application: {url}")
    print(f"Health Check: {url}health")
    print(f"Model Status: {url}model-status")
    print(f"API Status: {url}api-status")
    print(f"Database: PostgreSQL (verified & ready)")
    print(f"Unicode: Configured for scientific documents")
    print(f"Request Tracking: Enabled with UUID logging")
    print("=" * 60)

    info_id("Flask application ready to serve requests", startup_request_id)
    info_id("Request ID tracking enabled for all operations", startup_request_id)

    app.run(host=host, port=port, debug=debug_mode)