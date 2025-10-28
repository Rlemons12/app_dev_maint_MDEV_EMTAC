# modules/system_resources/system_resource_manager.py
import os
import psutil
import time
import threading
import sqlite3
from flask import current_app, g, request
from modules.configuration.log_config import info_id, warning_id, debug_id, error_id, with_request_id, \
    log_timed_operation
from modules.configuration.config_env import DatabaseConfig


class SystemResourceManager:
    """
    A utility class for managing and optimizing system resources in a Flask application with SQLite.

    This class provides methods to monitor and optimize CPU, memory, disk, database connections,
    and web request performance. It integrates with the existing logging and database
    connection management infrastructure.
    """

    # Class-level singleton database config instance
    _db_config = None

    # Class-level variables for monitoring
    _request_times_lock = threading.Lock()
    _request_times = []
    _resource_stats_history = []

    @classmethod
    def get_db_config(cls):
        """Get or create the database configuration singleton."""
        if cls._db_config is None:
            cls._db_config = DatabaseConfig()
        return cls._db_config

    @staticmethod
    @with_request_id
    def calculate_optimal_workers(memory_threshold=0.5, max_workers=None, request_id=None):
        """
        Calculate the optimal number of worker processes/threads based on available system memory.

        This method helps prevent memory exhaustion by limiting worker count based on available 
        memory and a safety threshold. It's useful for configuring thread pools, multiprocessing
        pools, or Gunicorn worker processes.

        Args:
            memory_threshold (float): Safety factor to limit memory usage (0.0-1.0)
            max_workers (int): Upper limit for number of workers, defaults to CPU count
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            int: Recommended number of worker processes/threads

        Example:
            workers = SystemResourceManager.calculate_optimal_workers(0.7)
            executor = ThreadPoolExecutor(max_workers=workers)
        """
        import psutil
        import os

        available_memory = psutil.virtual_memory().available
        memory_per_thread = 100 * 1024 * 1024  # Example: assume each thread uses 100MB
        max_memory_workers = available_memory // memory_per_thread

        if max_workers is None:
            max_workers = os.cpu_count()

        # Limit workers based on memory and CPU availability
        optimal_workers = min(max_memory_workers, max_workers)

        # Apply a memory threshold to avoid using all available memory
        result = max(1, int(optimal_workers * memory_threshold))

        info_id(f"Calculated optimal workers: {result} (available memory: {available_memory / (1024 * 1024):.2f} MB, "
                f"max workers: {max_workers}, memory threshold: {memory_threshold})", request_id)

        return result

    @staticmethod
    @with_request_id
    def optimize_sqlite_performance(db_path, request_id=None):
        """
        Apply performance-optimizing PRAGMA settings to a SQLite database.

        This method configures SQLite for better performance by setting appropriate journal mode,
        synchronization level, cache size, and other pragmas. It complements the existing
        settings applied in DatabaseConfig.

        Args:
            db_path (str): Path to the SQLite database file
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            bool: True if optimizations were successfully applied, False otherwise
        """
        try:
            with log_timed_operation("SQLite optimization", request_id):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Enhanced performance optimizations beyond what's in DatabaseConfig
                pragmas = [
                    "PRAGMA journal_mode = WAL",  # Write-Ahead Logging
                    "PRAGMA synchronous = NORMAL",  # Reduced synchronous mode for better performance
                    "PRAGMA cache_size = -64000",  # 64MB cache
                    "PRAGMA temp_store = MEMORY",  # Store temp tables in memory
                    "PRAGMA mmap_size = 30000000000",  # Memory mapping
                    "PRAGMA foreign_keys = ON",  # Always good practice
                    "PRAGMA analysis_limit = 1000",  # Limit for ANALYZE
                    "PRAGMA optimize",  # Run the query optimizer
                ]

                for pragma in pragmas:
                    cursor.execute(pragma)

                # Run ANALYZE to optimize query planning
                cursor.execute("ANALYZE")

                # Check for missing indices
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]
                    info_id(f"Analyzing table: {table_name}", request_id)

                    # Get columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()

                    # Get existing indices
                    cursor.execute(f"PRAGMA index_list({table_name})")
                    indices = {row[1]: row[0] for row in cursor.fetchall()}

                    # Check for foreign keys without indices
                    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                    fks = cursor.fetchall()

                    for fk in fks:
                        col_name = fk[3]
                        if f"idx_{table_name}_{col_name}" not in indices:
                            info_id(f"Missing index on foreign key: {table_name}.{col_name}", request_id)

                conn.commit()
                conn.close()
                info_id(f"SQLite optimizations applied to {db_path}", request_id)
                return True
        except Exception as e:
            error_id(f"Failed to optimize SQLite: {str(e)}", request_id)
            return False

    @classmethod
    @with_request_id
    def audit_db_connections(cls, threshold=80, request_id=None):
        """
        Audit database connections and verify they're being properly managed.

        This method checks for connection leaks by comparing the tracked connection count 
        with actual open file handles, and validates connection limiting settings.

        Args:
            threshold (int): Warning threshold percentage (0-100)
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            dict: Connection audit information
        """
        # Get the database config
        db_config = cls.get_db_config()

        # Get connection stats from DatabaseConfig
        stats = db_config.get_connection_stats()

        # Check file descriptors
        proc = psutil.Process()
        open_files = proc.open_files()
        sqlite_files = [f for f in open_files if f.path.endswith('.db') or '/tmp/sqlite' in f.path]

        # Get limits
        soft_limit, hard_limit = proc.rlimit(psutil.RLIMIT_NOFILE)
        fd_percent = (len(open_files) / soft_limit) * 100

        # Check for connection leaks (difference between tracked and actual connections)
        tracked_connections = stats['active_main_connections'] + stats['active_revision_connections']
        actual_sqlite_connections = len(sqlite_files)

        leak_detected = actual_sqlite_connections > tracked_connections
        if leak_detected:
            warning_id(f"Possible connection leak: tracked={tracked_connections}, actual={actual_sqlite_connections}",
                       request_id)

        # Check if we're approaching file descriptor limits
        if fd_percent > threshold:
            warning_id(f"File descriptor usage high: {fd_percent:.1f}% ({len(open_files)}/{soft_limit})",
                       request_id)

        # Compile audit results
        audit_result = {
            **stats,
            'open_file_descriptors': len(open_files),
            'file_descriptor_limit': soft_limit,
            'fd_percent_used': fd_percent,
            'actual_sqlite_connections': actual_sqlite_connections,
            'tracked_connections': tracked_connections,
            'possible_connection_leak': leak_detected
        }

        info_id(f"Database connection audit completed: {fd_percent:.1f}% FD used, "
                f"{actual_sqlite_connections} SQLite connections", request_id)

        return audit_result

    @classmethod
    @with_request_id
    def monitor_flask_performance(cls, app=None, slow_threshold=1.0, request_id=None):
        """
        Set up enhanced request timing monitoring for a Flask application.

        This method adds before_request and after_request handlers to track execution time
        of each request, collecting statistics for performance analysis. It integrates with
        the existing request_id middleware from log_config.

        Args:
            app (Flask): The Flask application instance
            slow_threshold (float): Time in seconds above which requests are considered slow
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            function: A callable that returns the latest performance statistics
        """
        if app is None:
            app = current_app

        # Clear any existing monitoring data
        with cls._request_times_lock:
            cls._request_times = []

        @app.before_request
        def track_request_start():
            # Store additional timing info
            g.resource_manager_start_time = time.time()
            g.resource_manager_start_memory = psutil.Process().memory_info().rss

        @app.after_request
        def track_request_end(response):
            if hasattr(g, 'resource_manager_start_time'):
                # Calculate metrics
                duration = time.time() - g.resource_manager_start_time
                memory_delta = 0

                if hasattr(g, 'resource_manager_start_memory'):
                    memory_delta = psutil.Process().memory_info().rss - g.resource_manager_start_memory

                # Store request info with timestamp for time-series analysis
                with cls._request_times_lock:
                    cls._request_times.append({
                        'timestamp': time.time(),
                        'path': request.path,
                        'method': request.method,
                        'status_code': response.status_code,
                        'duration': duration,
                        'memory_delta': memory_delta
                    })

                    # Keep only the most recent 1000 requests
                    if len(cls._request_times) > 1000:
                        cls._request_times.pop(0)

                # Add timing header
                response.headers['X-Response-Time'] = f"{duration:.6f}"

                # Log slow requests
                if duration > slow_threshold:
                    warning_id(f"Slow request: {request.method} {request.path} ({duration:.4f}s)",
                               request.id if hasattr(request, 'id') else None)

            return response

        def get_performance_stats(window_seconds=60):
            """Get performance statistics for recent requests."""
            now = time.time()
            cutoff = now - window_seconds

            with cls._request_times_lock:
                # Filter to recent requests
                recent_requests = [r for r in cls._request_times if r['timestamp'] > cutoff]

                if not recent_requests:
                    return {
                        'request_count': 0,
                        'avg_duration': 0,
                        'max_duration': 0,
                        'p95_duration': 0,
                        'requests_per_second': 0,
                        'status_codes': {},
                        'paths': {}
                    }

                # Calculate statistics
                durations = [r['duration'] for r in recent_requests]
                durations.sort()

                # Count status codes
                status_codes = {}
                for r in recent_requests:
                    status = r['status_code']
                    status_codes[status] = status_codes.get(status, 0) + 1

                # Count paths
                paths = {}
                for r in recent_requests:
                    path = r['path']
                    paths[path] = paths.get(path, 0) + 1

                # Calculate p95 (95th percentile)
                p95_index = int(len(durations) * 0.95)
                p95 = durations[p95_index] if p95_index < len(durations) else durations[-1]

                return {
                    'request_count': len(recent_requests),
                    'avg_duration': sum(durations) / len(durations) if durations else 0,
                    'max_duration': max(durations) if durations else 0,
                    'p95_duration': p95,
                    'requests_per_second': len(recent_requests) / window_seconds,
                    'status_codes': status_codes,
                    'paths': paths
                }

        info_id(f"Flask performance monitoring enabled (slow threshold: {slow_threshold}s)", request_id)
        return get_performance_stats

    @classmethod
    @with_request_id
    def monitor_system_resources(cls, interval=60, history_size=60, request_id=None):
        """
        Start a background thread to monitor system resources (CPU, memory, disk, network).

        This method periodically collects system performance metrics and stores them
        for later analysis. It helps identify resource constraints and performance issues.

        Args:
            interval (int): Monitoring interval in seconds
            history_size (int): Number of data points to keep in history
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            function: A callable that returns the latest resource statistics
        """
        # Store resource usage over time
        cls._resource_stats_history = []

        # Create a background thread for monitoring
        stop_event = threading.Event()

        def monitor_thread():
            while not stop_event.is_set():
                try:
                    # Collect system metrics
                    stats = {
                        'timestamp': time.time(),
                        'cpu': {
                            'percent': psutil.cpu_percent(interval=0.5),
                            'count': psutil.cpu_count(),
                            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
                        },
                        'memory': {
                            'total': psutil.virtual_memory().total,
                            'available': psutil.virtual_memory().available,
                            'percent': psutil.virtual_memory().percent
                        },
                        'disk': {
                            'usage': {path: psutil.disk_usage(path)._asdict()
                                      for path in psutil.disk_partitions(all=False)}
                        },
                        'network': {
                            'connections': len(psutil.net_connections()),
                            'io_counters': {k: v._asdict() for k, v in psutil.net_io_counters(pernic=True).items()}
                        },
                        'process': {
                            'count': len(psutil.pids()),
                            'current': {
                                'cpu_percent': psutil.Process().cpu_percent(),
                                'memory_percent': psutil.Process().memory_percent(),
                                'threads': psutil.Process().num_threads(),
                                'open_files': len(psutil.Process().open_files())
                            }
                        }
                    }

                    # Add database connection stats
                    db_config = cls.get_db_config()
                    stats['database'] = db_config.get_connection_stats()

                    # Store in history, with lock for thread safety
                    with cls._request_times_lock:  # Reuse the same lock
                        cls._resource_stats_history.append(stats)

                        # Keep history size limited
                        while len(cls._resource_stats_history) > history_size:
                            cls._resource_stats_history.pop(0)

                    # Check for resource constraints and log warnings
                    if stats['memory']['percent'] > 90:
                        warning_id(f"High memory usage: {stats['memory']['percent']}%", request_id)

                    if stats['cpu']['percent'] > 90:
                        warning_id(f"High CPU usage: {stats['cpu']['percent']}%", request_id)

                    for path, usage in stats['disk']['usage'].items():
                        if usage['percent'] > 90:
                            warning_id(f"High disk usage on {path}: {usage['percent']}%", request_id)

                except Exception as e:
                    error_id(f"Error in resource monitoring thread: {str(e)}", request_id)

                # Sleep until next interval
                stop_event.wait(interval)

        # Start the monitoring thread
        threading.Thread(target=monitor_thread, daemon=True).start()

        def get_resource_stats(window_seconds=None):
            """Get resource statistics for recent interval."""
            with cls._request_times_lock:
                if not cls._resource_stats_history:
                    return None

                if window_seconds is None:
                    # Return the most recent stats
                    return cls._resource_stats_history[-1]

                # Filter to the window
                now = time.time()
                cutoff = now - window_seconds
                window_stats = [s for s in cls._resource_stats_history if s['timestamp'] > cutoff]

                if not window_stats:
                    return None

                # Return the latest plus some aggregate info
                latest = window_stats[-1]
                cpu_avg = sum(s['cpu']['percent'] for s in window_stats) / len(window_stats)
                mem_avg = sum(s['memory']['percent'] for s in window_stats) / len(window_stats)

                return {
                    'latest': latest,
                    'window_size': len(window_stats),
                    'cpu_avg': cpu_avg,
                    'memory_avg': mem_avg
                }

        # Function to stop monitoring
        def stop_monitoring():
            stop_event.set()

        # Store stop function as attribute for later cleanup
        get_resource_stats.stop = stop_monitoring

        info_id(f"System resource monitoring started (interval: {interval}s)", request_id)
        return get_resource_stats

    @staticmethod
    @with_request_id
    def analyze_slow_queries(db_path, threshold=0.1, request_id=None):
        """
        Analyze and identify slow SQLite queries by enabling query timing.

        This method creates a custom connection factory that logs slow queries,
        helping identify and optimize performance bottlenecks in database access.

        Args:
            db_path (str): Path to the SQLite database file
            threshold (float): Time in seconds above which queries are considered slow
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            function: A connection factory function to use for monitored connections
        """
        # Dictionary to store query statistics
        query_stats = {}
        query_stats_lock = threading.Lock()

        # Create a subclass of sqlite3.Connection to track query times
        class MonitoredConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                start = time.time()
                try:
                    result = super().execute(sql, *args, **kwargs)
                    elapsed = time.time() - start

                    # Store query statistics
                    with query_stats_lock:
                        if sql not in query_stats:
                            query_stats[sql] = {
                                'count': 0,
                                'total_time': 0,
                                'min_time': float('inf'),
                                'max_time': 0,
                                'slow_count': 0
                            }

                        stats = query_stats[sql]
                        stats['count'] += 1
                        stats['total_time'] += elapsed
                        stats['min_time'] = min(stats['min_time'], elapsed)
                        stats['max_time'] = max(stats['max_time'], elapsed)

                        if elapsed > threshold:
                            stats['slow_count'] += 1

                    # Log slow queries
                    if elapsed > threshold:
                        # Truncate long queries for logging
                        truncated_sql = sql[:500] + "..." if len(sql) > 500 else sql
                        warning_id(f"Slow SQLite query ({elapsed:.4f}s): {truncated_sql}", request_id)

                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    error_id(f"SQLite query error after {elapsed:.4f}s: {str(e)}\nQuery: {sql}", request_id)
                    raise

        # Function to get query statistics
        def get_query_stats():
            with query_stats_lock:
                return {sql: stats.copy() for sql, stats in query_stats.items()}

        # Return a factory function that creates monitored connections
        def connection_factory():
            conn = sqlite3.connect(db_path, factory=MonitoredConnection, detect_types=sqlite3.PARSE_DECLTYPES)
            return conn

        # Attach the stats function to the factory
        connection_factory.get_stats = get_query_stats

        info_id(f"SQLite query monitoring enabled for {db_path} with {threshold}s threshold", request_id)
        return connection_factory

    @staticmethod
    @with_request_id
    def optimize_static_files(app=None, max_age=86400, request_id=None):
        """
        Configure optimal HTTP headers for static file serving in Flask.

        This method adds appropriate cache headers to static assets (CSS, JS, images)
        to improve page load times for returning visitors and reduce server load.

        Args:
            app (Flask): The Flask application instance
            max_age (int): Cache lifetime in seconds (default: 1 day)
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            bool: True if static file optimization was successfully configured
        """
        if app is None:
            app = current_app

        @app.after_request
        def add_cache_headers(response):
            # Only apply to static files
            if response.mimetype.startswith(('text/css', 'application/javascript', 'image/')):
                response.cache_control.public = True
                response.cache_control.max_age = max_age
                response.headers['Vary'] = 'Accept-Encoding'

                # Add etag or Last-Modified header
                if not response.headers.get('ETag') and not response.headers.get('Last-Modified'):
                    import hashlib
                    response.set_etag(hashlib.sha1(response.data).hexdigest())

            return response

        info_id(f"Static file optimization configured with {max_age}s max age", request_id)
        return True

    @staticmethod
    @with_request_id
    def configure_rate_limiting(app=None, default_limits="200 per minute", request_id=None):
        """
        Set up rate limiting for Flask routes to prevent abuse and overload.

        This method adds a decorator that can be applied to Flask routes to limit
        the number of requests from a single client. It helps prevent DoS attacks
        and ensures fair resource allocation.

        Args:
            app (Flask): The Flask application instance
            default_limits (str): Default rate limit string (e.g., "100 per minute")
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            function: A decorator to apply rate limiting to routes

        Note:
            Requires Flask-Limiter package ('pip install Flask-Limiter')
        """
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
        except ImportError:
            error_id("Flask-Limiter not installed. Run 'pip install Flask-Limiter'", request_id)
            return None

        if app is None:
            app = current_app

        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[default_limits]
        )

        info_id(f"Rate limiting configured with default limit: {default_limits}", request_id)

        # Return the limiter's decorator for use on routes
        return limiter.limit

    @staticmethod
    @with_request_id
    def health_check(request_id=None):
        """
        Perform a comprehensive health check of the application and system resources.

        This method checks CPU, memory, disk, database connections, and other vital
        signs to determine if the application is healthy. It's useful for monitoring
        tools and load balancers.

        Args:
            request_id (str): Optional ID for tracking this operation in logs

        Returns:
            dict: Health check results with status indicators
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }

        # Check CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            health['checks']['cpu'] = {
                'status': 'healthy' if cpu_percent < 90 else 'warning',
                'percent': cpu_percent
            }

            if cpu_percent >= 90:
                health['status'] = 'degraded'
        except Exception as e:
            health['checks']['cpu'] = {'status': 'error', 'message': str(e)}
            health['status'] = 'degraded'

        # Check memory
        try:
            memory = psutil.virtual_memory()
            health['checks']['memory'] = {
                'status': 'healthy' if memory.percent < 90 else 'warning',
                'percent': memory.percent,
                'available_mb': memory.available / (1024 * 1024)
            }

            if memory.percent >= 90:
                health['status'] = 'degraded'
        except Exception as e:
            health['checks']['memory'] = {'status': 'error', 'message': str(e)}
            health['status'] = 'degraded'

        # Check disk
        try:
            disk = psutil.disk_usage('/')
            health['checks']['disk'] = {
                'status': 'healthy' if disk.percent < 90 else 'warning',
                'percent': disk.percent,
                'free_gb': disk.free / (1024 * 1024 * 1024)
            }

            if disk.percent >= 90:
                health['status'] = 'degraded'
        except Exception as e:
            health['checks']['disk'] = {'status': 'error', 'message': str(e)}
            health['status'] = 'degraded'

        # Check database
        try:
            # Create a test connection to verify database access
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
            session.execute("SELECT 1")
            session.close()

            # Get connection stats
            stats = db_config.get_connection_stats()
            conn_percent = (stats['active_main_connections'] / stats['max_concurrent_connections']) * 100

            health['checks']['database'] = {
                'status': 'healthy' if conn_percent < 80 else 'warning',
                'connections': stats['active_main_connections'],
                'connection_percent': conn_percent
            }

            if conn_percent >= 80:
                health['status'] = 'degraded'
        except Exception as e:
            health['checks']['database'] = {'status': 'error', 'message': str(e)}
            health['status'] = 'critical'  # Database failure is critical

        # Log overall health status
        if health['status'] == 'healthy':
            info_id("Health check passed: All systems operational", request_id)
        elif health['status'] == 'degraded':
            warning_id("Health check warning: System performance degraded", request_id)
        else:
            error_id("Health check failed: System in critical condition", request_id)

        return health