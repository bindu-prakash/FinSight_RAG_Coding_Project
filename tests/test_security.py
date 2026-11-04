
# ================================================================
# FINSIGHT RAG - SECURITY UNIT TESTS
# ================================================================
# This file tests the authentication-related document access
# control functions defined in src/security.py.
#
# Main objectives:
# 1. Verify that admins can access all documents.
# 2. Verify that unknown documents are denied to normal users.
# 3. Verify that document permissions are correctly enforced.
# 4. Verify that unauthorized documents are filtered out
#    before the retrieval process.
#
# Testing framework: pytest
# ================================================================


# ------------------------------------------------
# 1. IMPORT REQUIRED FUNCTIONS
# ------------------------------------------------

from src.security import can_access, filter_documents_for_role
# Imports two functions from the security module.
#
# can_access():
# Checks whether a particular user role can access a document.
#
# filter_documents_for_role():
# Filters a list of documents based on the user's role.
#
# These are the functions that will be tested in this file.


# ================================================================
# 2. TEST ADMIN ACCESS
# ================================================================

def test_admin_can_access_unknown_document():
    # Tests whether an admin can access a document
    # that is not explicitly listed in the permission dictionary.

    assert can_access("new.pdf", "admin", {}) is True
    # assert:
    # Pytest uses assert to check whether a condition is true.
    #
    # "new.pdf":
    # Name of the document being requested.
    #
    # "admin":
    # Role of the current user.
    #
    # {}:
    # Empty permission dictionary.
    #
    # Expected result:
    # True, because admins are allowed to access
    # all indexed documents in this prototype.
    #
    # If the function returns False, this test fails.


# ================================================================
# 3. TEST DEFAULT DENIAL FOR UNKNOWN DOCUMENTS
# ================================================================

def test_unknown_document_denied_by_default():
    # Tests the fail-closed security behaviour.
    #
    # A document that is not present in the permission dictionary
    # should not be accessible to non-admin users.

    assert can_access("new.pdf", "analyst", {}) is False
    # Checks whether an analyst is denied access
    # to an unknown document.
    #
    # Expected result: False.

    assert can_access("new.pdf", "viewer", {}) is False
    # Checks whether a viewer is denied access
    # to the same unknown document.
    #
    # Expected result: False.
    #
    # These assertions ensure that access is denied
    # unless permission is explicitly granted.


# ================================================================
# 4. TEST ALLOWED AND DENIED ROLES
# ================================================================

def test_allowed_roles():
    # Tests whether document-specific role permissions
    # are correctly checked.

    perms = {
        "report.pdf": {"admin", "analyst"}
    }
    # Creates a sample permission dictionary.
    #
    # report.pdf:
    # The document being protected.
    #
    # {"admin", "analyst"}:
    # Only admins and analysts are allowed to access this document.
    #
    # Viewer is not included in the allowed roles.

    assert can_access("report.pdf", "analyst", perms) is True
    # Checks whether an analyst can access report.pdf.
    #
    # Expected result: True.
    #
    # Analyst is included in the allowed roles.

    assert can_access("report.pdf", "viewer", perms) is False
    # Checks whether a viewer can access report.pdf.
    #
    # Expected result: False.
    #
    # Viewer is not included in the allowed roles.
    #
    # This verifies that document permissions are role-specific.


# ================================================================
# 5. TEST DOCUMENT FILTERING BEFORE RETRIEVAL
# ================================================================

def test_filter_before_retrieval():
    # Tests whether unauthorized documents are removed
    # before they are passed to the retrieval stage.

    docs = [
        {
            "source": "allowed.pdf",
            "text": "public"
        },
        {
            "source": "secret.pdf",
            "text": "restricted"
        }
    ]
    # Creates a sample list containing two documents.
    #
    # allowed.pdf:
    # A document containing public information.
    #
    # secret.pdf:
    # A document containing restricted information.
    #
    # Each document is represented as a Python dictionary.
    #
    # source:
    # Stores the document filename.
    #
    # text:
    # Stores the document content.

    perms = {
        "allowed.pdf": {"analyst"}
    }
    # Creates the access permission dictionary.
    #
    # Only analysts are allowed to access allowed.pdf.
    #
    # secret.pdf is not listed.
    # Therefore, non-admin users must be denied access to it.

    filtered = filter_documents_for_role(
        docs,
        "analyst",
        perms
    )
    # Calls filter_documents_for_role().
    #
    # docs:
    # Complete list of available documents.
    #
    # "analyst":
    # Role of the current user.
    #
    # perms:
    # Document access permissions.
    #
    # The function checks each document and returns
    # only the documents that the analyst is permitted to access.
    #
    # Expected filtered result:
    # [
    #     {
    #         "source": "allowed.pdf",
    #         "text": "public"
    #     }
    # ]
    #
    # secret.pdf should be excluded.

    assert [
        d["source"] for d in filtered
    ] == ["allowed.pdf"]
    # Verifies the final list of accessible document filenames.
    #
    # List comprehension:
    # Extracts the source filename from every filtered document.
    #
    # Expected result:
    # ["allowed.pdf"]
    #
    # This confirms that:
    # 1. The permitted document is retained.
    # 2. The restricted document is removed.
    # 3. Filtering occurs before document retrieval.


# ================================================================
# END OF SECURITY TEST FILE
# ================================================================
