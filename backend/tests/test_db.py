"""Test helpers for models using an in-memory SQLModel database.

These tests provide a quick sanity check that the CRUD helpers in
`backend.models` work as expected for the paired questionnaire design.

Run with:
    pytest backend/test_db.py
"""

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from backend.models import (
    create_generated_response_pair,
    create_likert_question,
    create_likert_response,
    create_prompt_pair,
    create_survey_item,
    create_survey_item_response,
    create_survey_submission,
    create_user,
    delete_generated_response_pair,
    delete_likert_question,
    delete_likert_response,
    delete_prompt_pair,
    delete_survey_item,
    delete_survey_item_response,
    delete_survey_submission,
    delete_user,
    get_generated_response_pair,
    get_likert_question,
    get_likert_response,
    get_prompt_pair,
    get_survey_item,
    get_survey_item_response,
    get_survey_submission,
    get_user,
    get_user_by_email,
    list_generated_response_pairs,
    list_likert_questions,
    list_likert_responses,
    list_prompt_pairs,
    list_survey_item_responses,
    list_survey_items,
    list_survey_submissions,
    list_users,
    update_generated_response_pair,
    update_likert_question,
    update_likert_response,
    update_prompt_pair,
    update_survey_item,
    update_survey_item_response,
    update_survey_submission,
    update_user_email,
)


def _create_in_memory_session():
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


# ============================================================================
# USER TESTS
# ============================================================================

def test_create_user():
    """Test creating a new user."""
    session = _create_in_memory_session()
    user = create_user(session, email="alice@example.com")
    
    assert user.id is not None
    assert user.email == "alice@example.com"


def test_get_user():
    """Test retrieving a user by ID."""
    session = _create_in_memory_session()
    user = create_user(session, email="bob@example.com")
    
    retrieved = get_user(session, user.id)
    assert retrieved is not None
    assert retrieved.id == user.id
    assert retrieved.email == "bob@example.com"


def test_get_user_not_found():
    """Test that getting a non-existent user returns None."""
    session = _create_in_memory_session()
    user = get_user(session, 999)
    assert user is None


def test_get_user_by_email():
    """Test retrieving a user by email."""
    session = _create_in_memory_session()
    user = create_user(session, email="charlie@example.com")
    
    retrieved = get_user_by_email(session, "charlie@example.com")
    assert retrieved is not None
    assert retrieved.id == user.id


def test_get_user_by_email_not_found():
    """Test that getting a non-existent user by email returns None."""
    session = _create_in_memory_session()
    user = get_user_by_email(session, "nonexistent@example.com")
    assert user is None


def test_list_users():
    """Test listing all users."""
    session = _create_in_memory_session()
    create_user(session, email="user1@example.com")
    create_user(session, email="user2@example.com")
    create_user(session, email="user3@example.com")
    
    users = list_users(session)
    assert len(users) == 3


def test_update_user_email():
    """Test updating a user's email."""
    session = _create_in_memory_session()
    user = create_user(session, email="old@example.com")
    
    updated = update_user_email(session, user.id, "new@example.com")
    assert updated is not None
    assert updated.email == "new@example.com"
    
    # Verify the change persisted
    retrieved = get_user(session, user.id)
    assert retrieved.email == "new@example.com"


def test_delete_user():
    """Test deleting a user."""
    session = _create_in_memory_session()
    user = create_user(session, email="todelete@example.com")
    
    deleted = delete_user(session, user.id)
    assert deleted is True
    
    retrieved = get_user(session, user.id)
    assert retrieved is None


def test_delete_user_not_found():
    """Test deleting a non-existent user returns False."""
    session = _create_in_memory_session()
    deleted = delete_user(session, 999)
    assert deleted is False


# ============================================================================
# PROMPT PAIR TESTS
# ============================================================================

def test_create_prompt_pair():
    """Test creating a new prompt pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(
        session,
        category="Algorithm",
        round=1,
        aave_text="Can you explain dis algorithm to me?",
        sae_text="Can you explain this algorithm to me?",
        display_order=1
    )
    
    assert prompt.id is not None
    assert prompt.category == "Algorithm"
    assert prompt.round == 1
    assert prompt.aave_text == "Can you explain dis algorithm to me?"
    assert prompt.sae_text == "Can you explain this algorithm to me?"
    assert prompt.display_order == 1


def test_get_prompt_pair():
    """Test retrieving a prompt pair by ID."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(
        session, "ELI5", 1, "Explain AI simple like", "Explain AI simply"
    )
    
    retrieved = get_prompt_pair(session, prompt.id)
    assert retrieved is not None
    assert retrieved.category == "ELI5"


def test_list_prompt_pairs():
    """Test listing prompt pairs with filters."""
    session = _create_in_memory_session()
    create_prompt_pair(session, "Cat1", 1, "aave1", "sae1")
    create_prompt_pair(session, "Cat2", 1, "aave2", "sae2")
    create_prompt_pair(session, "Cat1", 2, "aave3", "sae3")
    
    # List all
    all_pairs = list_prompt_pairs(session)
    assert len(all_pairs) == 3
    
    # Filter by round
    round1 = list_prompt_pairs(session, round=1)
    assert len(round1) == 2
    
    # Filter by category
    cat1 = list_prompt_pairs(session, category="Cat1")
    assert len(cat1) == 2


def test_update_prompt_pair():
    """Test updating a prompt pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Category", 1, "old aave", "old sae")
    
    updated = update_prompt_pair(
        session, prompt.id, aave_text="new aave", display_order=5
    )
    assert updated.aave_text == "new aave"
    assert updated.display_order == 5
    assert updated.category == "Category"  # unchanged


def test_delete_prompt_pair():
    """Test deleting a prompt pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    
    deleted = delete_prompt_pair(session, prompt.id)
    assert deleted is True
    
    retrieved = get_prompt_pair(session, prompt.id)
    assert retrieved is None


# ============================================================================
# GENERATED RESPONSE PAIR TESTS
# ============================================================================

def test_create_generated_response_pair():
    """Test creating a generated response pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    
    response = create_generated_response_pair(
        session,
        prompt_pair_id=prompt.id,
        method="API",
        model="GPT-4",
        aave_response="This be da response in AAVE",
        sae_response="This is the response in SAE"
    )
    
    assert response.id is not None
    assert response.prompt_pair_id == prompt.id
    assert response.method == "API"
    assert response.model == "GPT-4"


def test_list_generated_response_pairs():
    """Test listing response pairs with filters."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    
    create_generated_response_pair(
        session, prompt.id, "method1", "model1", "aave1", "sae1"
    )
    create_generated_response_pair(
        session, prompt.id, "method2", "model1", "aave2", "sae2"
    )
    
    all_pairs = list_generated_response_pairs(session)
    assert len(all_pairs) == 2
    
    method1_pairs = list_generated_response_pairs(session, method="method1")
    assert len(method1_pairs) == 1


def test_update_generated_response_pair():
    """Test updating a generated response pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    response = create_generated_response_pair(
        session, prompt.id, "method", "model", "old aave", "old sae"
    )
    
    updated = update_generated_response_pair(
        session, response.id, model="GPT-4o"
    )
    assert updated.model == "GPT-4o"


def test_delete_generated_response_pair():
    """Test deleting a generated response pair."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    response = create_generated_response_pair(
        session, prompt.id, "method", "model", "aave", "sae"
    )
    
    deleted = delete_generated_response_pair(session, response.id)
    assert deleted is True


# ============================================================================
# SURVEY ITEM TESTS
# ============================================================================

def test_create_survey_item():
    """Test creating a survey item."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    
    item = create_survey_item(
        session,
        round=1,
        category="Cat",
        prompt_pair_id=prompt.id,
        display_order=1,
        is_active=True
    )
    
    assert item.id is not None
    assert item.prompt_pair_id == prompt.id
    assert item.is_active is True


def test_list_survey_items():
    """Test listing survey items with filters."""
    session = _create_in_memory_session()
    prompt1 = create_prompt_pair(session, "Cat1", 1, "aave", "sae")
    prompt2 = create_prompt_pair(session, "Cat2", 2, "aave", "sae")
    
    create_survey_item(session, 1, "Cat1", prompt1.id, is_active=True)
    create_survey_item(session, 1, "Cat1", prompt1.id, is_active=False)
    create_survey_item(session, 2, "Cat2", prompt2.id, is_active=True)
    
    # Filter by round
    round1 = list_survey_items(session, round=1)
    assert len(round1) == 2
    
    # Filter by active status
    active = list_survey_items(session, is_active=True)
    assert len(active) == 2


def test_update_survey_item():
    """Test updating a survey item."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id, is_active=True)
    
    updated = update_survey_item(session, item.id, is_active=False)
    assert updated.is_active is False


def test_delete_survey_item():
    """Test deleting a survey item."""
    session = _create_in_memory_session()
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    
    deleted = delete_survey_item(session, item.id)
    assert deleted is True


# ============================================================================
# LIKERT QUESTION TESTS
# ============================================================================

def test_create_likert_question():
    """Test creating a likert question."""
    session = _create_in_memory_session()
    question = create_likert_question(
        session,
        question="How natural does this sound?",
        dimension="naturalness",
        scale_min=1,
        scale_max=5
    )
    
    assert question.id is not None
    assert question.dimension == "naturalness"
    assert question.scale_min == 1
    assert question.scale_max == 5


def test_list_likert_questions():
    """Test listing likert questions."""
    session = _create_in_memory_session()
    create_likert_question(session, "Q1", "naturalness")
    create_likert_question(session, "Q2", "clarity")
    create_likert_question(session, "Q3", "naturalness")
    
    all_questions = list_likert_questions(session)
    assert len(all_questions) == 3
    
    naturalness = list_likert_questions(session, dimension="naturalness")
    assert len(naturalness) == 2


def test_update_likert_question():
    """Test updating a likert question."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Old question", "naturalness")
    
    updated = update_likert_question(
        session, question.id, question="New question"
    )
    assert updated.question == "New question"


def test_delete_likert_question():
    """Test deleting a likert question."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Q", "dim")
    
    deleted = delete_likert_question(session, question.id)
    assert deleted is True


# ============================================================================
# SURVEY SUBMISSION TESTS
# ============================================================================

def test_create_survey_submission():
    """Test creating a survey submission."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    
    submission = create_survey_submission(
        session,
        user_id=user.id,
        round=1,
        short_answer="This is my answer"
    )
    
    assert submission.id is not None
    assert submission.user_id == user.id
    assert submission.round == 1
    assert submission.short_answer == "This is my answer"


def test_get_survey_submission():
    """Test retrieving a survey submission."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    submission = create_survey_submission(session, user.id, 1)
    
    retrieved = get_survey_submission(session, submission.id)
    assert retrieved is not None
    assert retrieved.user_id == user.id


def test_list_survey_submissions():
    """Test listing survey submissions."""
    session = _create_in_memory_session()
    user1 = create_user(session, "user1@example.com")
    user2 = create_user(session, "user2@example.com")
    
    create_survey_submission(session, user1.id, 1)
    create_survey_submission(session, user1.id, 2)
    create_survey_submission(session, user2.id, 1)
    
    user1_submissions = list_survey_submissions(session, user_id=user1.id)
    assert len(user1_submissions) == 2
    
    round1_submissions = list_survey_submissions(session, round=1)
    assert len(round1_submissions) == 2


def test_update_survey_submission():
    """Test updating a survey submission."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    submission = create_survey_submission(session, user.id, 1)
    
    updated = update_survey_submission(
        session, submission.id, short_answer="Updated answer"
    )
    assert updated.short_answer == "Updated answer"


def test_delete_survey_submission():
    """Test deleting a survey submission."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    submission = create_survey_submission(session, user.id, 1)
    
    deleted = delete_survey_submission(session, submission.id)
    assert deleted is True


# ============================================================================
# SURVEY ITEM RESPONSE TESTS
# ============================================================================

def test_create_survey_item_response():
    """Test creating a survey item response."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    
    response = create_survey_item_response(
        session,
        submission.id,
        item.id,
        []
    )
    
    assert response.id is not None
    assert response.submission_id == submission.id
    assert response.survey_item_id == item.id


def test_list_survey_item_responses():
    """Test listing survey item responses."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item1 = create_survey_item(session, 1, "Cat", prompt.id)
    item2 = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    
    create_survey_item_response(session, submission.id, item1.id, [])
    create_survey_item_response(session, submission.id, item2.id, [])
    
    responses = list_survey_item_responses(session, submission_id=submission.id)
    assert len(responses) == 2


def test_update_survey_item_response():
    """Test updating a survey item response."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    response = create_survey_item_response(session, submission.id, item.id, [])
    
    updated = update_survey_item_response(session, response.id, likert_answers=[])
    assert updated.survey_item_id == item.id


def test_delete_survey_item_response():
    """Test deleting a survey item response."""
    session = _create_in_memory_session()
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    response = create_survey_item_response(session, submission.id, item.id, [])
    
    deleted = delete_survey_item_response(session, response.id)
    assert deleted is True


# ============================================================================
# LIKERT RESPONSE TESTS
# ============================================================================

def test_create_likert_response():
    """Test creating a likert response."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Is this natural?", "naturalness")
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    item_response = create_survey_item_response(session, submission.id, item.id, [])
    
    likert_resp = create_likert_response(
        session,
        item_response.id,
        question.id,
        4,
        "Pretty natural"
    )
    
    assert likert_resp.id is not None
    assert likert_resp.value == 4
    assert likert_resp.comment == "Pretty natural"


def test_list_likert_responses():
    """Test listing likert responses."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Q", "dim")
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    item_response = create_survey_item_response(session, submission.id, item.id, [])
    
    create_likert_response(session, item_response.id, question.id, 3)
    create_likert_response(session, item_response.id, question.id, 4)
    
    responses = list_likert_responses(session, item_response_id=item_response.id)
    assert len(responses) == 2


def test_update_likert_response():
    """Test updating a likert response."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Q", "dim")
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    item_response = create_survey_item_response(session, submission.id, item.id, [])
    likert_resp = create_likert_response(session, item_response.id, question.id, 2)
    
    updated = update_likert_response(session, likert_resp.id, value=5)
    assert updated.value == 5


def test_delete_likert_response():
    """Test deleting a likert response."""
    session = _create_in_memory_session()
    question = create_likert_question(session, "Q", "dim")
    user = create_user(session, "user@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    item = create_survey_item(session, 1, "Cat", prompt.id)
    submission = create_survey_submission(session, user.id, 1)
    item_response = create_survey_item_response(session, submission.id, item.id, [])
    likert_resp = create_likert_response(session, item_response.id, question.id, 3)
    
    deleted = delete_likert_response(session, likert_resp.id)
    assert deleted is True


