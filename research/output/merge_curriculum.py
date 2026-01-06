#!/usr/bin/env python3
"""
Merge Era 4-5 content into curriculum-full.json
Normalizes different schemas to match Era 1-3 format
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(os.path.join(BASE_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

def normalize_era4_topic(topic, quiz_data):
    """Convert Era 4 topic to standard format with quiz embedded"""
    # Find matching quiz
    quiz = None
    for q in quiz_data.get('quizzes', []):
        if q['topicSlug'] == topic['slug']:
            quiz = {
                'title': q['title'],
                'passingScore': q['passingScore'],
                'isGate': q['isGate'],
                'questions': q['questions']
            }
            break

    return {
        'slug': topic['slug'],
        'title': topic['title'],
        'description': topic['description'],
        'era': 'deep-learning',
        'linearOrder': topic['linearOrder'],
        'icon': topic['icon'],
        'estimatedMinutes': topic['estimatedMinutes'],
        'lessons': topic['lessons'],
        'quiz': quiz
    }

def normalize_era5_topic(topic, quiz_data):
    """Convert Era 5 topic to standard format with quiz embedded"""
    # Convert lessons to standard format
    lessons = []
    for lesson in topic.get('lessons', []):
        lessons.append({
            'slug': lesson['slug'],
            'title': lesson['title'],
            'contentPath': lesson['file'],  # 'file' -> 'contentPath'
            'lessonOrder': lesson['order'],
            'lessonType': 'content'
        })

    # Find and convert matching quiz
    quiz = None
    for q in quiz_data.get('quizzes', []):
        if q['topicSlug'] == topic['slug']:
            # Convert questions to standard format
            questions = []
            for i, question in enumerate(q['questions']):
                # Convert index-based correctAnswer to text-based
                correct_idx = question['correctAnswer']
                correct_text = question['options'][correct_idx]

                questions.append({
                    'questionText': question['question'],  # 'question' -> 'questionText'
                    'questionType': 'multiple_choice',
                    'options': question['options'],
                    'correctAnswer': correct_text,
                    'explanation': question['explanation'],
                    'questionOrder': i + 1
                })

            quiz = {
                'title': f"{topic['title']} Knowledge Check",
                'passingScore': 70,
                'isGate': True,
                'questions': questions
            }
            break

    return {
        'slug': topic['slug'],
        'title': topic['title'],
        'description': topic['description'],
        'era': 'modern-ai',
        'linearOrder': topic['linearOrder'],
        'icon': 'cpu',  # Default icon for Era 5
        'estimatedMinutes': 40,  # Default
        'lessons': lessons,
        'quiz': quiz
    }

def normalize_era4_connections(conn_data):
    """Era 4 connections are already in the correct format"""
    return conn_data.get('connections', [])

def normalize_era5_connections(conn_data):
    """Convert Era 5 connections from source/target to fromTopicSlug/toTopicSlug"""
    connections = []
    for conn in conn_data.get('connections', []):
        # Map type names
        type_map = {
            'requires': 'leads_to',
            'enabled': 'enabled',
            'enables': 'enabled',
            'influences': 'influenced',
            'foundation_for': 'enabled',
            'leads_to': 'leads_to',
            'conceptual_link': 'conceptual_link',
            'preceded': 'preceded'
        }

        connections.append({
            'fromTopicSlug': conn['source'],
            'toTopicSlug': conn['target'],
            'connectionType': type_map.get(conn['type'], conn['type']),
            'label': conn['label']
        })
    return connections

def main():
    print("Loading curriculum-full.json (Eras 1-3)...")
    curriculum = load_json('curriculum-full.json')

    print("Loading Era 4 data...")
    era4_topics = load_json('era4-topics.json')
    era4_quizzes = load_json('era4-quiz-data.json')
    era4_connections = load_json('era4-connections.json')

    print("Loading Era 5 data...")
    era5_topics = load_json('era5-topics.json')
    era5_quizzes = load_json('era5-quiz-data.json')
    era5_connections = load_json('era5-connections.json')

    print(f"Current topics: {len(curriculum['topics'])}")
    print(f"Current connections: {len(curriculum['connections'])}")

    # Add Era 4 topics
    print("\nAdding Era 4 topics...")
    for topic in era4_topics['topics']:
        normalized = normalize_era4_topic(topic, era4_quizzes)
        curriculum['topics'].append(normalized)
        print(f"  Added: {topic['slug']}")

    # Add Era 5 topics
    print("\nAdding Era 5 topics...")
    for topic in era5_topics['topics']:
        normalized = normalize_era5_topic(topic, era5_quizzes)
        curriculum['topics'].append(normalized)
        print(f"  Added: {topic['slug']}")

    # Add Era 4 connections
    print("\nAdding Era 4 connections...")
    era4_conns = normalize_era4_connections(era4_connections)
    curriculum['connections'].extend(era4_conns)
    print(f"  Added {len(era4_conns)} connections")

    # Add Era 5 connections
    print("\nAdding Era 5 connections...")
    era5_conns = normalize_era5_connections(era5_connections)
    curriculum['connections'].extend(era5_conns)
    print(f"  Added {len(era5_conns)} connections")

    # Update totals
    print(f"\nFinal topics: {len(curriculum['topics'])}")
    print(f"Final connections: {len(curriculum['connections'])}")

    # Save merged curriculum
    print("\nSaving merged curriculum...")
    save_json(curriculum, 'curriculum-full.json')
    print("Done!")

if __name__ == '__main__':
    main()
