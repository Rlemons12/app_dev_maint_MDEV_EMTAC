@chatbot_bp.route('/rate', methods=['POST'])
def rate_question():
    try:
        data = request.json
        user_id = data.get('userId')
        qanda_id = data.get('qandaId')
        rating = data.get('rating')
        feedback = data.get('feedback')

        # Create a new Rating instance and add it to the database
        new_rating = Rating(user_id=user_id, qanda_id=qanda_id, rating=rating, feedback=feedback)
        with LocalSession() as session:
            session.add(new_rating)
            session.commit()

        return jsonify({'message': 'Rating submitted successfully'})

    except SQLAlchemyError as e:
        LocalSession.rollback()
        logger.error(f"Database error: {e}")
        return jsonify({'error': 'Database error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
