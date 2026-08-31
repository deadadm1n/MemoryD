from memoryd.entities import EntityAssociation, extract_entities, extract_entity_associations


def test_extract_entities_canonicalizes_and_keeps_first_appearance_order():
    text = "Atlas Gateway uses Postgres with Python. SQLite remains a local fallback; postgres is deferred."

    assert extract_entities(text) == ["Atlas Gateway", "PostgreSQL", "Python", "SQLite"]


def test_extract_entities_filters_generic_words_and_handles_empty_values():
    assert extract_entities("The project and the system use this API.") == []
    assert extract_entities("") == []
    assert extract_entities(None) == []


def test_extract_entity_associations_are_sentence_scoped_and_unique():
    text = "memoryd connects SQLite and Model Context Protocol. Python and SQLite are used separately. SQLite and MCP are linked."

    assert extract_entity_associations(text) == [
        EntityAssociation("SQLite", "Model Context Protocol"),
        EntityAssociation("Python", "SQLite"),
    ]
