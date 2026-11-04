
# ================================================================
# FINSIGHT RAG - SECURITY AND ACCESS CONTROL
# ================================================================
# This file manages authentication and document-level authorization.
#
# Main responsibilities:
# 1. Define the user roles.
# 2. Load registered user information.
# 3. Verify usernames and passwords securely.
# 4. Load document access permissions.
# 5. Check whether a user can access a document.
# 6. Filter documents based on the logged-in user's role.
#
# Supported roles:
# - Admin
# - Analyst
# - Viewer
#
# This module is imported by app.py to control user access.
# ================================================================


# ------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------

import hashlib
# hashlib:
# Provides cryptographic hashing functions.
#
# In this project, PBKDF2-HMAC-SHA256 is used
# to verify passwords without storing plain-text passwords.


import hmac
# hmac:
# Provides Hash-based Message Authentication Code functions.
#
# compare_digest() is used to compare password hashes
# in a way designed to reduce timing-based information leakage.


import json
# json:
# Used to read JSON configuration files.
#
# User account information is stored in users.json.


from pathlib import Path
# Path:
# Provides an object-oriented way to work with file paths.
#
# It is used to locate and read configuration files.


import csv
# csv:
# Used to read CSV files.
#
# Document-level access permissions are stored
# in document_permissions.csv.


# ================================================================
# 2. DEFINE USER ROLES
# ================================================================

ROLES = {"admin", "analyst", "viewer"}
# Defines the three supported user roles.
#
# admin:
# Has access to all indexed documents in this prototype.
#
# analyst:
# Can access documents explicitly assigned to the analyst role.
#
# viewer:
# Can access documents explicitly assigned to the viewer role.
#
# A Python set is used because it stores unique values
# and supports efficient membership checking.


# ================================================================
# 3. LOAD USER INFORMATION
# ================================================================

def load_users(path="config/users.json"):
    # Loads registered user account information from a JSON file.
    #
    # path:
    # Location of the users.json configuration file.
    #
    # The default path is config/users.json.

    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    # Path(path):
    # Converts the supplied file path into a Path object.
    #
    # read_text():
    # Reads the contents of the JSON file as text.
    #
    # encoding="utf-8":
    # Ensures the file is read using UTF-8 encoding.
    #
    # json.loads():
    # Converts the JSON text into a Python dictionary.
    #
    # The returned dictionary contains the registered users
    # and their stored authentication information.


# ================================================================
# 4. VERIFY USERNAME AND PASSWORD
# ================================================================

def verify_password(username, password, users):
    # Checks whether the entered username and password are valid.
    #
    # username:
    # Username entered by the user.
    #
    # password:
    # Password entered by the user.
    #
    # users:
    # Dictionary containing registered user information.
    #
    # Returns:
    # A dictionary containing username and role if authentication
    # succeeds; otherwise, returns None.

    user = users.get(username)
    # get():
    # Retrieves the account information for the given username.
    #
    # If the username does not exist, get() returns None.
    #
    # user:
    # Stores the matching user's account information.

    if not user:
        # Checks whether the username was not found.

        return None
        # Stops authentication if the account does not exist.


    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(user["salt"]),
        180000
    ).hex()
    # Generates a password hash using PBKDF2-HMAC-SHA256.
    #
    # hashlib.pbkdf2_hmac():
    # Applies the PBKDF2 password-based key derivation function
    # using HMAC as the pseudorandom function.
    #
    # "sha256":
    # Specifies SHA-256 as the hashing algorithm.
    #
    # password.encode("utf-8"):
    # Converts the entered password from a string into bytes.
    #
    # bytes.fromhex(user["salt"]):
    # Converts the user's stored hexadecimal salt into bytes.
    #
    # A salt is a random value used during password hashing.
    # It helps prevent identical passwords from producing
    # identical stored hashes.
    #
    # 180000:
    # Specifies the number of PBKDF2 iterations.
    # More iterations increase the computational cost
    # of password guessing.
    #
    # .hex():
    # Converts the generated hash bytes into a hexadecimal string.
    #
    # actual:
    # Stores the newly calculated hash of the entered password.


    if hmac.compare_digest(
        actual,
        user["password_hash"]
    ):
        # Compares the calculated hash with the stored password hash.
        #
        # hmac.compare_digest():
        # Performs a timing-resistant comparison.
        #
        # If both hashes match, the entered password is correct.

        return {
            "username": username,
            "role": user["role"]
        }
        # Authentication is successful.
        #
        # Returns the username and assigned role.
        #
        # app.py can use this information to identify
        # the logged-in user and control access.


    return None
    # If the hashes do not match, authentication fails.
    # None indicates an invalid username/password combination.


# ================================================================
# 5. LOAD DOCUMENT PERMISSIONS
# ================================================================

def load_document_permissions(
    path="config/document_permissions.csv"
):
    # Loads document-level access permissions from a CSV file.
    #
    # path:
    # Location of document_permissions.csv.
    #
    # The file specifies which roles can access each document.
    #
    # Returns:
    # A dictionary mapping document names to allowed roles.

    permissions = {}
    # Creates an empty dictionary to store document permissions.
    #
    # Example structure:
    # {
    #     "annual_report.pdf": {"admin", "analyst"},
    #     "summary.pdf": {"admin", "viewer"}
    # }


    with open(
        path,
        newline="",
        encoding="utf-8"
    ) as f:
        # open():
        # Opens the CSV file.
        #
        # path:
        # File location.
        #
        # newline="":
        # Recommended when reading CSV files.
        #
        # encoding="utf-8":
        # Reads the file using UTF-8 encoding.
        #
        # as f:
        # Stores the opened file object in the variable f.
        #
        # with:
        # Ensures the file is properly closed after processing.


        for row in csv.DictReader(f):
            # csv.DictReader():
            # Reads CSV rows as dictionaries.
            #
            # The CSV column headings become dictionary keys.
            #
            # Example:
            # {
            #     "source_name": "annual_report.pdf",
            #     "allowed_roles": "admin;analyst"
            # }
            #
            # row:
            # Represents the current CSV record.


            permissions[row["source_name"]] = {
                role.strip()
                for role in row["allowed_roles"].split(";")
                if role.strip()
            }
            # row["source_name"]:
            # Gets the document filename.
            # This filename becomes the dictionary key.
            #
            # row["allowed_roles"]:
            # Gets the roles assigned to that document.
            #
            # .split(";"):
            # Splits the role string wherever a semicolon appears.
            #
            # Example:
            # "admin;analyst;viewer"
            # becomes:
            # ["admin", "analyst", "viewer"]
            #
            # role.strip():
            # Removes unnecessary spaces around each role.
            #
            # if role.strip():
            # Ignores empty role values.
            #
            # Set comprehension:
            # Creates a set containing the allowed roles.
            #
            # The final permission entry maps one document
            # to its set of permitted roles.


    return permissions
    # Returns the completed document permission dictionary.


# ================================================================
# 6. CHECK DOCUMENT ACCESS
# ================================================================

def can_access(source_name, role, permissions):
    # Checks whether a particular user role can access a document.
    #
    # source_name:
    # Name of the document being requested.
    #
    # role:
    # Role assigned to the logged-in user.
    #
    # permissions:
    # Dictionary containing document access permissions.
    #
    # Returns:
    # True if access is allowed.
    # False if access is denied.

    # Admins may manage/access all indexed documents
    # in this local prototype.

    if role == "admin":
        # Checks whether the current user's role is admin.

        return True
        # Admin users are allowed to access every indexed document.


    # Fail closed:
    # Unknown documents are denied to non-admin users.

    return role in permissions.get(source_name, set())
    # permissions.get(source_name, set()):
    # Retrieves the allowed roles for the requested document.
    #
    # If the document is not listed, an empty set is returned.
    #
    # role in ...:
    # Checks whether the user's role belongs to the allowed roles.
    #
    # Returns True if the role is permitted.
    # Returns False if the role is not permitted.
    #
    # This follows the fail-closed security principle:
    # access is denied unless permission is explicitly granted.


# ================================================================
# 7. FILTER DOCUMENTS BASED ON USER ROLE
# ================================================================

def filter_documents_for_role(
    documents,
    role,
    permissions
):
    # Filters the loaded document list according to user permissions.
    #
    # documents:
    # List of all loaded documents.
    #
    # role:
    # Role of the logged-in user.
    #
    # permissions:
    # Document permission dictionary.
    #
    # Returns:
    # Only the documents that the current role is allowed to access.

    return [
        d
        for d in documents
        if can_access(
            d["source"],
            role,
            permissions
        )
    ]
    # List comprehension:
    # Creates a new list containing only permitted documents.
    #
    # for d in documents:
    # Iterates through all loaded documents.
    #
    # d["source"]:
    # Retrieves the filename of the current document.
    #
    # can_access():
    # Calls the access-checking function.
    #
    # Only documents for which can_access() returns True
    # are included in the final list.
    #
    # This prevents unauthorized documents from being
    # passed to the later document-processing and retrieval stages.


# ================================================================
# END OF SECURITY MODULE
# ================================================================
