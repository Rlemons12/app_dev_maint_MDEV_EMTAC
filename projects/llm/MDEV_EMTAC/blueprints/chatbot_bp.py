import logging
import os
import sys
import time
import re
import uuid
from modules.emtac_ai.aist_manager import AistManager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Blueprint, request, jsonify, current_app, url_for, redirect
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import (QandA, ChatSession, KeywordSearch, CompleteDocument)
from datetime import datetime
from plugins.ai_modules.ai_models import ModelsConfig
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (logger, with_request_id, log_timed_operation, debug_id,
                                              info_id, error_id, warning_id)

# Global AistManager instance for reuse across requests
global_aist_manager = None

# Create blueprint
chatbot_bp = Blueprint('chatbot_bp', __name__)


# Complete corrected function:
def get_or_create_aist_manager():
    """
    FIXED: Get or create a global AistManager instance WITH database session for tracking.
    """
    global global_aist_manager

    if global_aist_manager is None:
        try:
            logger.info(" Creating AistManager with tracking support...")

            # CRITICAL FIX: Get database session for tracking
            db_config = DatabaseConfig()
            db_session = db_config.get_main_session()  #  FIXED: Use get_main_session()

            if not db_session:
                logger.error(" Could not get database session")
                # Create without session as fallback
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(ai_model=ai_model)
            else:
                logger.info(" Database session obtained")

                # Load AI model using ModelsConfig
                ai_model = ModelsConfig.load_ai_model()

                # FIXED: Create AistManager with database session
                global_aist_manager = AistManager(
                    ai_model=ai_model,
                    db_session=db_session  # CRITICAL: Pass database session during initialization
                )

            logger.info(" Global AistManager created successfully")

            # Test tracking initialization
            if hasattr(global_aist_manager, 'check_tracking_status'):
                tracking_status = global_aist_manager.check_tracking_status()
                if tracking_status.get('tracking_enabled', False):
                    logger.info(" Search tracking is fully operational")
                else:
                    logger.warning(" Search tracking is not operational")
                    logger.info(f" Tracking status: {tracking_status}")
            else:
                logger.warning(" No tracking status check method available")

        except Exception as e:
            logger.error(f" Failed to create AistManager with tracking: {e}")
            # Fallback without session
            try:
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(ai_model=ai_model)
                logger.info(" Created fallback AistManager without tracking")
            except Exception as fallback_error:
                logger.error(f" Fallback AistManager creation failed: {fallback_error}")
                global_aist_manager = AistManager()

    return global_aist_manager



@chatbot_bp.route('/update_qanda', methods=['POST'])
@with_request_id
def update_qanda(request_id=None):
    """Update Q&A entries with ratings and comments."""
    start_time = time.time()

    info_id(f"Received request to update Q&A", request_id)

    try:
        # Parse request data
        user_id = request.json.get('user_id', 'anonymous')
        question = request.json.get('question', '')
        answer = request.json.get('answer', '')
        rating = request.json.get('rating')
        comment = request.json.get('comment')

        debug_id(
            f"Update data - user_id: {user_id}, rating: {rating}, comment length: {len(str(comment)) if comment else 0}",
            request_id)

        # Create database session using DatabaseConfig
        local_session = None
        db_config = None

        try:
            with log_timed_operation("create_db_session", request_id):
                db_config = DatabaseConfig()
                local_session = db_config.get_main_session()
                debug_id(f"Database session created for Q&A update", request_id)

            with log_timed_operation("update_qa_entry", request_id):
                # Check if QandA has the update_or_create_feedback method
                if hasattr(QandA, 'update_or_create_feedback'):
                    success = QandA.update_or_create_feedback(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        rating=rating,
                        comment=comment,
                        session=local_session
                    )
                else:
                    # Fallback to manual update if method doesn't exist
                    last_qanda_entry = local_session.query(QandA).order_by(QandA.id.desc()).first()

                    if last_qanda_entry and last_qanda_entry.rating is None and last_qanda_entry.comment is None:
                        last_qanda_entry.rating = rating
                        last_qanda_entry.comment = comment
                        success = True
                    else:
                        new_qanda = QandA(
                            user_id=user_id,
                            question=question,
                            answer=answer,
                            rating=rating,
                            comment=comment,
                            timestamp=datetime.now().isoformat()
                        )
                        local_session.add(new_qanda)
                        success = True

                if success:
                    local_session.commit()
                    info_id(f"Q&A updated successfully in {time.time() - start_time:.3f}s", request_id)
                    return jsonify({'message': 'Q&A updated successfully'})
                else:
                    local_session.rollback()
                    warning_id(f"Failed to update Q&A entry", request_id)
                    return jsonify({'error': 'Failed to update Q&A entry'}), 400

        except SQLAlchemyError as e:
            if local_session:
                local_session.rollback()
            error_id(f"Database error while updating Q&A: {e}", request_id, exc_info=True)
            return jsonify({'error': str(e)}), 500
        finally:
            if local_session:
                local_session.close()
                debug_id(f"Database session closed", request_id)

    except Exception as e:
        error_id(f"Unexpected error in update_qanda: {e}", request_id, exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

@chatbot_bp.route('/ask', methods=['POST'])
@with_request_id
def ask(request_id=None):
    """Process questions and return answers using AistManager with detailed performance monitoring."""

    # Ensure all responses include the 4 block containers
    def _ensure_blocks(response: dict) -> dict:
        if "blocks" not in response or not isinstance(response["blocks"], dict):
            response["blocks"] = {}
        for key in ("parts-container", "images-container", "documents-container", "drawings-container"):
            response["blocks"].setdefault(key, [])
        return response

    performance_metrics = {
        'request_start': time.time(),
        'steps': {},
        'total_time': 0,
        'step_count': 0
    }

    def track_step(step_name, step_start_time, end_time=None):
        if end_time is None:
            end_time = time.time()
        duration = end_time - step_start_time
        performance_metrics['steps'][step_name] = {
            'duration': duration,
            'start_offset': step_start_time - performance_metrics['request_start']
        }
        performance_metrics['step_count'] += 1

        if duration > 1.0:
            warning_id(f"Slow step '{step_name}': {duration:.3f}s", request_id)
        elif duration > 0.5:
            info_id(f"Step '{step_name}' took {duration:.3f}s", request_id)
        else:
            debug_id(f"Step '{step_name}' completed in {duration:.3f}s", request_id)

        return duration

    local_session = None
    db_config = None

    info_id("New chat request received", request_id)

    try:
        # --- Step 1: Input parsing ---
        step_start = time.time()
        data = request.json or {}
        debug_id(f"Request data keys: {list(data.keys()) if data else 'None'}", request_id)

        user_id = data.get('userId', 'anonymous')
        question = data.get('question', '').strip()
        client_type = data.get('clientType', 'web')
        rating = data.get('rating')
        comment = data.get('comment')
        track_step('input_parsing', step_start)

        info_id(f"Processing question from user {user_id}: '{question[:100]}{'...' if len(question) > 100 else ''}'", request_id)
        debug_id(f"Client type: {client_type}, Rating: {rating}, Comment present: {bool(comment)}", request_id)

        if not question or len(question) < 3:
            warning_id(f"Question too short: '{question}'", request_id)
            performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
            return jsonify(_ensure_blocks({
                'answer': "Please provide a more detailed question so I can help you better.",
                'status': 'invalid_input',
                'performance': performance_metrics
            }))

        if question.lower() == "end session please":
            info_id(f"User {user_id} requested to end the session", request_id)
            performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
            return jsonify(_ensure_blocks({
                'answer': "Session ended. Thank you for using the chatbot!",
                'status': 'session_ended',
                'redirect': url_for('logout_bp.logout'),
                'performance': performance_metrics
            }))

        # --- Step 2: Init AistManager ---
        step_start = time.time()
        try:
            aist_manager = get_or_create_aist_manager()
            debug_id("AistManager obtained", request_id)
            track_step('aist_manager_init', step_start)
        except Exception as manager_err:
            track_step('aist_manager_init_failed', step_start)
            error_id(f"Failed to get AistManager: {manager_err}", request_id, exc_info=True)
            performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
            return jsonify(_ensure_blocks({
                'answer': "I'm having trouble initializing the AI system. Please try again.",
                'status': 'manager_error',
                'performance': performance_metrics
            })), 500

        # --- Step 3: Database session ---
        step_start = time.time()
        try:
            db_config = DatabaseConfig()
            local_session = db_config.get_main_session()
            debug_id("Database session created", request_id)
            aist_manager.db_session = local_session
            track_step('database_session_creation', step_start)
        except Exception as db_err:
            track_step('database_session_creation_failed', step_start)
            error_id(f"Failed to create database session: {db_err}", request_id, exc_info=True)
            performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
            return jsonify(_ensure_blocks({
                'answer': "I'm having trouble accessing the database. Please try again.",
                'status': 'database_error',
                'performance': performance_metrics
            })), 500

        # --- Step 4: Process with AistManager ---
        try:
            step_start = time.time()
            aist_manager.begin_request(request_id)
            track_step('aist_manager_begin_request', step_start)

            step_start = time.time()
            info_id("Sending question to AistManager for processing", request_id)
            result = aist_manager.answer_question(
                question=question,
                user_id=user_id,
                request_id=request_id,
                client_type=client_type
            )
            processing_time = track_step('aist_manager_processing', step_start)

            if not isinstance(result, dict):
                error_id(f"AistManager returned invalid format: {type(result)}", request_id)
                result = {'status': 'error', 'answer': 'Invalid response format', 'method': 'error_recovery'}

            status = result.get('status')

            # --- Step 5: Format outputs ---
            step_start = time.time()

            if status == 'end_session':
                info_id("AistManager requested session end", request_id)
                performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
                return jsonify(_ensure_blocks({
                    'answer': "Session ended as requested.",
                    'status': 'session_ended',
                    'redirect': url_for('logout_bp.logout'),
                    'performance': performance_metrics
                }))

            elif status == 'success':
                answer = result.get('answer', 'No answer provided')
                method = result.get('method', 'unknown')
                total_time = time.time() - performance_metrics['request_start']
                performance_metrics['total_time'] = total_time

                info_id(f"Question answered using '{method}' strategy, answer length: {len(answer)}", request_id)

                response_data = {
                    'answer': answer,
                    'status': 'success',
                    'method': method,
                    'response_time': total_time,
                    'performance': performance_metrics if client_type == 'debug' else {
                        'total_time': total_time,
                        'processing_time': processing_time,
                        'method': method
                    }
                }
                return jsonify(_ensure_blocks({**result, **response_data}))

            elif status == 'error':
                error_msg = result.get('answer', "I encountered an issue.")
                error_id(f"AistManager returned error: {error_msg}", request_id)
                performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
                return jsonify(_ensure_blocks({
                    'answer': error_msg,
                    'status': 'error',
                    'performance': performance_metrics
                })), 500

            else:
                warning_id(f"Unknown status from AistManager: {status}", request_id)
                performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
                return jsonify(_ensure_blocks({
                    'answer': result.get('answer', "Unexpected response format."),
                    'status': status or 'unknown',
                    'method': result.get('method', 'unknown'),
                    'performance': performance_metrics
                })), 200

        except Exception as processing_err:
            if local_session:
                try:
                    local_session.rollback()
                except:
                    pass
            error_id(f"Error during AistManager processing: {processing_err}", request_id, exc_info=True)
            performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
            return jsonify(_ensure_blocks({
                'answer': "I encountered an unexpected issue while processing your question.",
                'status': 'processing_error',
                'error_type': type(processing_err).__name__,
                'performance': performance_metrics
            })), 500

        finally:
            if local_session:
                try:
                    local_session.close()
                    debug_id("Database session closed", request_id)
                except Exception as cleanup_err:
                    error_id(f"Error during cleanup: {cleanup_err}", request_id)
            track_step('cleanup', time.time())

    except SQLAlchemyError as e:
        error_id(f"Database error: {e}", request_id, exc_info=True)
        performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
        return jsonify(_ensure_blocks({
            'answer': "I'm having trouble accessing the information you need right now. Please try again later.",
            'status': 'database_error',
            'performance': performance_metrics
        })), 500

    except Exception as e:
        error_id(f"Unexpected error in ask route: {e}", request_id, exc_info=True)
        performance_metrics['total_time'] = time.time() - performance_metrics['request_start']
        return jsonify(_ensure_blocks({
            'answer': "I encountered an unexpected issue. Please try rephrasing your question.",
            'status': 'unexpected_error',
            'performance': performance_metrics
        })), 500

@chatbot_bp.route('/health', methods=['GET'])
@with_request_id
def health_check(request_id=None):
    """Health check endpoint for the chatbot service."""
    try:
        # Check if AistManager can be created
        aist_manager = get_or_create_aist_manager()

        # Check database connectivity
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        session.execute("SELECT 1")
        session.close()

        # Check AI model status
        ai_model_name = ModelsConfig.get_current_ai_model_name()

        info_id(f"Health check passed", request_id)
        return jsonify({
            'status': 'healthy',
            'aist_manager': 'available',
            'database': 'connected',
            'ai_model': ai_model_name,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        error_id(f"Health check failed: {e}", request_id, exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@chatbot_bp.route('/reset_session', methods=['POST'])
@with_request_id
def reset_session(request_id=None):
    """Reset the chat session for a user."""
    try:
        data = request.json
        user_id = data.get('userId', 'anonymous')

        info_id(f"Resetting session for user {user_id}", request_id)

        # Get AistManager and handle session reset
        aist_manager = get_or_create_aist_manager()

        # Create temporary session for the reset operation
        db_config = DatabaseConfig()
        local_session = db_config.get_main_session()

        try:
            aist_manager.db_session = local_session

            # Process reset through AistManager
            result = aist_manager.answer_question(user_id, "reset context", 'api')

            if result.get('status') == 'success':
                info_id(f"Session reset successfully for user {user_id}", request_id)
                return jsonify({
                    'message': 'Session reset successfully',
                    'status': 'success'
                })
            else:
                warning_id(f"Session reset failed for user {user_id}", request_id)
                return jsonify({
                    'message': 'Failed to reset session',
                    'status': 'failed'
                }), 500

        finally:
            aist_manager.db_session = None
            local_session.close()

    except Exception as e:
        error_id(f"Error resetting session: {e}", request_id, exc_info=True)
        return jsonify({
            'error': 'Failed to reset session',
            'status': 'error'
        }), 500


@chatbot_bp.route('/performance', methods=['GET'])
@with_request_id
def performance_stats(request_id=None):
    """Get performance statistics for the chatbot service."""
    try:
        # Get performance data from the global AistManager
        aist_manager = get_or_create_aist_manager()

        # Basic system performance check
        start_time = time.time()

        # Database performance test
        db_start = time.time()
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        session.execute("SELECT 1")
        session.close()
        db_time = time.time() - db_start

        # AI model performance test (if available)
        ai_start = time.time()
        ai_model_name = "Not Available"
        ai_response_time = 0
        try:
            if aist_manager.ai_model:
                ai_model_name = type(aist_manager.ai_model).__name__
                # Quick test response
                test_response = aist_manager.ai_model.get_response("Test")
                ai_response_time = time.time() - ai_start
        except Exception as ai_err:
            debug_id(f"AI model test failed: {ai_err}", request_id)
            ai_response_time = -1

        total_time = time.time() - start_time

        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'database': {
                'response_time': db_time,
                'status': 'healthy' if db_time < 0.5 else 'slow' if db_time < 2.0 else 'unhealthy'
            },
            'ai_model': {
                'name': ai_model_name,
                'response_time': ai_response_time,
                'status': 'healthy' if 0 < ai_response_time < 2.0 else 'slow' if ai_response_time < 5.0 else 'unhealthy'
            },
            'system': {
                'total_check_time': total_time,
                'aist_manager_status': 'available' if aist_manager else 'unavailable'
            },
            'thresholds': {
                'database_warning': 0.5,
                'database_critical': 2.0,
                'ai_model_warning': 2.0,
                'ai_model_critical': 5.0
            }
        }

        info_id(f"Performance check completed in {total_time:.3f}s", request_id)
        return jsonify(performance_data)

    except Exception as e:
        error_id(f"Performance check failed: {e}", request_id, exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@chatbot_bp.route('/metrics', methods=['GET'])
@with_request_id
def metrics(request_id=None):
    """Get detailed metrics about recent chatbot performance."""
    try:
        # Get recent Q&A entries for performance analysis
        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        try:
            # Get recent Q&A entries (last 100)
            recent_qas = session.query(QandA).order_by(QandA.id.desc()).limit(100).all()

            if not recent_qas:
                return jsonify({
                    'message': 'No recent Q&A data available',
                    'metrics': {},
                    'timestamp': datetime.now().isoformat()
                })

            # Calculate performance metrics
            total_interactions = len(recent_qas)
            rated_interactions = len([qa for qa in recent_qas if qa.rating is not None])
            avg_rating = sum(qa.rating for qa in recent_qas if
                             qa.rating is not None) / rated_interactions if rated_interactions > 0 else 0

            # Question length analysis
            avg_question_length = sum(len(qa.question) for qa in recent_qas) / total_interactions
            avg_answer_length = sum(len(qa.answer) for qa in recent_qas) / total_interactions

            # Model usage statistics (if model_name field exists)
            model_usage = {}
            if hasattr(QandA, 'model_name'):
                for qa in recent_qas:
                    model = getattr(qa, 'model_name', 'Unknown')
                    model_usage[model] = model_usage.get(model, 0) + 1

            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'sample_size': total_interactions,
                'interactions': {
                    'total': total_interactions,
                    'rated': rated_interactions,
                    'rating_percentage': (
                                rated_interactions / total_interactions * 100) if total_interactions > 0 else 0
                },
                'quality': {
                    'average_rating': round(avg_rating, 2),
                    'rating_distribution': {
                        '1': len([qa for qa in recent_qas if qa.rating == 1]),
                        '2': len([qa for qa in recent_qas if qa.rating == 2]),
                        '3': len([qa for qa in recent_qas if qa.rating == 3]),
                        '4': len([qa for qa in recent_qas if qa.rating == 4]),
                        '5': len([qa for qa in recent_qas if qa.rating == 5])
                    }
                },
                'content': {
                    'avg_question_length': round(avg_question_length, 1),
                    'avg_answer_length': round(avg_answer_length, 1)
                },
                'models': model_usage
            }

            info_id(f"Metrics calculated for {total_interactions} interactions", request_id)
            return jsonify(metrics_data)

        finally:
            session.close()

    except Exception as e:
        error_id(f"Metrics calculation failed: {e}", request_id, exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@chatbot_bp.route('/performance/recommendations', methods=['GET'])
@with_request_id
def performance_recommendations(request_id=None):
    """Get performance recommendations based on recent analytics."""
    try:
        aist_manager = get_or_create_aist_manager()

        # Check if the manager has the get_performance_analytics method
        if not hasattr(aist_manager, 'get_performance_analytics'):
            return jsonify({
                'message': 'Performance analytics not available - method not implemented',
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            })

        # Get analytics for the last 24 hours
        analytics = aist_manager.get_performance_analytics(hours=24)

        if not analytics or analytics.get('total_requests', 0) == 0:
            return jsonify({
                'message': 'No recent performance data available for analysis',
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            })

        recommendations = []

        # Check overall response time
        avg_time = analytics.get('avg_response_time', 0)
        if avg_time > 3.0:
            recommendations.append({
                'priority': 'high',
                'category': 'response_time',
                'message': f'Average response time is {avg_time:.2f}s, consider optimizing search strategies',
                'current_value': avg_time,
                'target_value': 2.0
            })

        # Check step performance
        step_perf = analytics.get('step_performance', {})

        if 'ai_response' in step_perf and step_perf['ai_response']['avg_time'] > 2.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'ai_model',
                'message': f'AI model responses averaging {step_perf["ai_response"]["avg_time"]:.2f}s, consider model optimization',
                'current_value': step_perf['ai_response']['avg_time'],
                'target_value': 2.0
            })

        if 'vector_search' in step_perf and step_perf['vector_search']['avg_time'] > 1.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'vector_search',
                'message': f'Vector search averaging {step_perf["vector_search"]["avg_time"]:.2f}s, consider index optimization',
                'current_value': step_perf['vector_search']['avg_time'],
                'target_value': 1.0
            })

        if 'fulltext_search' in step_perf and step_perf['fulltext_search']['avg_time'] > 1.0:
            recommendations.append({
                'priority': 'low',
                'category': 'fulltext_search',
                'message': f'Full-text search averaging {step_perf["fulltext_search"]["avg_time"]:.2f}s, consider database indexing',
                'current_value': step_perf['fulltext_search']['avg_time'],
                'target_value': 0.5
            })

        # Check performance distribution
        perf_dist = analytics.get('performance_distribution', {})
        total_requests = analytics['total_requests']
        poor_performance_ratio = (perf_dist.get('poor', 0) + perf_dist.get('very_poor', 0)) / total_requests

        if poor_performance_ratio > 0.2:  # More than 20% poor performance
            recommendations.append({
                'priority': 'high',
                'category': 'overall',
                'message': f'{poor_performance_ratio * 100:.1f}% of requests have poor performance, investigate bottlenecks',
                'current_value': poor_performance_ratio * 100,
                'target_value': 10.0
            })

        # Check method efficiency
        method_perf = analytics.get('method_performance', {})
        for method, stats in method_perf.items():
            if stats['avg_time'] > 4.0 and stats['count'] > 5:  # Only if we have enough samples
                recommendations.append({
                    'priority': 'medium',
                    'category': 'method_optimization',
                    'message': f'Method "{method}" averaging {stats["avg_time"]:.2f}s over {stats["count"]} requests',
                    'current_value': stats['avg_time'],
                    'target_value': 3.0
                })

        # Sort recommendations by priority
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        response_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_hours': 24,
            'total_requests_analyzed': total_requests,
            'overall_performance_score': analytics.get('avg_performance_score', 0),
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                'low_priority': len([r for r in recommendations if r['priority'] == 'low'])
            }
        }

        info_id(f"Generated {len(recommendations)} performance recommendations", request_id)
        return jsonify(response_data)

    except Exception as e:
        error_id(f"Performance recommendations failed: {e}", request_id, exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@chatbot_bp.route('/performance/dashboard', methods=['GET'])
@with_request_id
def performance_dashboard(request_id=None):
    """Get comprehensive performance dashboard data."""
    try:
        aist_manager = get_or_create_aist_manager()
        hours = request.args.get('hours', 24, type=int)

        # Initialize performance_data to handle missing method
        performance_data = {
            'total_requests': 0,
            'avg_response_time': 0,
            'avg_performance_score': 0
        }

        # Get current performance stats if method exists
        perf_start = time.time()
        if hasattr(aist_manager, 'get_performance_analytics'):
            try:
                performance_data = aist_manager.get_performance_analytics(hours=hours)
            except Exception as perf_err:
                debug_id(f"Performance analytics failed: {perf_err}", request_id)

        perf_check_time = time.time() - perf_start

        # Get system health
        health_start = time.time()
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        session.execute("SELECT 1")
        session.close()
        db_check_time = time.time() - health_start

        # AI model check
        ai_start = time.time()
        ai_status = "available" if aist_manager.ai_model else "unavailable"
        ai_model_name = type(aist_manager.ai_model).__name__ if aist_manager.ai_model else "None"
        ai_check_time = time.time() - ai_start

        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'system_health': {
                'database': {
                    'status': 'healthy' if db_check_time < 0.5 else 'slow',
                    'response_time': db_check_time
                },
                'ai_model': {
                    'status': ai_status,
                    'name': ai_model_name,
                    'check_time': ai_check_time
                },
                'aist_manager': {
                    'status': 'available' if aist_manager else 'unavailable',
                    'performance_check_time': perf_check_time
                }
            },
            'performance_analytics': performance_data,
            'quick_stats': {
                'total_requests': performance_data.get('total_requests', 0),
                'avg_response_time': performance_data.get('avg_response_time', 0),
                'performance_score': performance_data.get('avg_performance_score', 0)
            }
        }

        info_id(f"Performance dashboard generated for {hours}h period", request_id)
        return jsonify(dashboard_data)

    except Exception as e:
        error_id(f"Performance dashboard failed: {e}", request_id, exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# Initialize the global AistManager when the module is loaded
try:
    global_aist_manager = get_or_create_aist_manager()
    logger.info("Chatbot blueprint initialized with AistManager")
except Exception as init_err:
    logger.error(f"Failed to initialize AistManager during module load: {init_err}", exc_info=True)
    global_aist_manager = None