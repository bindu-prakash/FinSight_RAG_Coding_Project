
# ================================================================
# FINSIGHT RAG - MAIN APPLICATION FILE
# ================================================================
# This is the main entry-point file of our project.
#
# It creates the Streamlit website and connects:
# 1. User authentication
# 2. Role-Based Access Control (RBAC)
# 3. Financial document loading
# 4. Document chunking and embeddings
# 5. RAG-based question answering
# 6. LLM-only baseline comparison
#
# Programming language: Python
# User interface framework: Streamlit
# Embedding model: all-MiniLM-L6-v2
# Language model: Qwen2.5:3B
# Local LLM runtime: Ollama
# ================================================================


# ------------------------------------------------
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ------------------------------------------------

import sys
# sys is a built-in Python module.
# It provides access to Python interpreter-related functions.
# Here, we use sys.path to tell Python where to search for modules.


from pathlib import Path
# Path is imported from Python's pathlib module.
# It helps us create, access and manage file and folder paths.
# It works across operating systems such as Windows and Linux.


import streamlit as st
# Imports the Streamlit library.
# Streamlit is used to create interactive web applications using Python.
# 'as st' gives the library a shorter name.
# Instead of writing streamlit.title(), we can write st.title().


# ------------------------------------------------
# SECTION 2: DEFINING THE PROJECT ROOT DIRECTORY
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# ROOT stores the main project directory path.
#
# __file__:
# A special Python variable that represents the current file's path.
#
# Path():
# Converts the file path into a Path object.
#
# resolve():
# Converts the path into an absolute, resolved path.
#
# parents[1]:
# Moves two directory levels upward from the current file.
#
# Example project structure:
#
# FinSight_RAG_Coding_Project/
#       |
#       |---- config/
#       |---- data/
#       |---- src/
#              |---- app.py
#
# ROOT points to FinSight_RAG_Coding_Project.


sys.path.insert(0, str(ROOT / "src"))

# sys.path:
# A Python list containing directories where Python searches for modules.
#
# insert(0, ...):
# Adds a new directory at position zero.
# Position zero means Python checks this directory early.
#
# ROOT / "src":
# Creates the path to the src folder.
#
# str():
# Converts the Path object into a string.
#
# Purpose:
# This allows us to import Python modules stored inside src,
# such as security.py and rag_core.py.


# ------------------------------------------------
# SECTION 3: IMPORTING OUR PROJECT MODULES
# ------------------------------------------------

from security import load_users, verify_password, load_document_permissions, filter_documents_for_role

# security.py contains the security and authentication functions.
#
# load_users():
# Loads user account information from the users.json file.
#
# verify_password():
# Checks whether the entered username and password are valid.
#
# load_document_permissions():
# Loads document access permissions from the CSV file.
#
# filter_documents_for_role():
# Filters documents according to the logged-in user's role.
#
# These functions implement user authentication and RBAC.


from rag_core import load_documents, build_rows, embed_rows, answer_with_agent, generate_answer

# rag_core.py contains the main RAG processing functions.
#
# load_documents():
# Loads financial documents from the selected folder.
#
# build_rows():
# Processes document content and creates smaller text chunks.
# These chunks are also associated with metadata such as source and page.
#
# embed_rows():
# Converts the document chunks into numerical embeddings.
#
# answer_with_agent():
# Executes the RAG-based question-answering workflow.
#
# generate_answer():
# Generates an answer using the language model.
# It is also used for the LLM-only baseline comparison.


# ------------------------------------------------
# SECTION 4: CONFIGURING THE STREAMLIT WEBSITE
# ------------------------------------------------

st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📊",
    layout="wide"
)

# set_page_config():
# Configures the appearance and layout of the Streamlit page.
#
# page_title:
# Sets the title displayed in the browser tab.
#
# page_icon:
# Sets the icon displayed in the browser tab.
#
# layout="wide":
# Makes the application use a wider page layout.


st.title("📊 FinSight RAG")
# title():
# Displays the main heading of the website.


st.caption("Evidence-grounded financial annual-report QA • GenAI + bounded agent workflow")
# caption():
# Displays smaller descriptive text below the main title.
#
# Evidence-grounded:
# Answers are supported by information retrieved from documents.
#
# QA:
# Stands for Question Answering.
#
# GenAI:
# Stands for Generative Artificial Intelligence.
#
# Bounded agent workflow:
# A predefined sequence of steps that processes a question,
# retrieves evidence and generates an answer.


# ------------------------------------------------
# SECTION 5: LOADING USER CONFIGURATION
# ------------------------------------------------

@st.cache_data
def users_config():
    return load_users(str(ROOT / "config" / "users.json"))

# @st.cache_data:
# This is a Streamlit decorator.
# It stores the returned data in a cache.
# This helps avoid repeatedly loading the same user configuration.
#
# def:
# Python keyword used to define a function.
#
# users_config():
# A function created to load user account information.
#
# return:
# Sends the result back to the part of the program that called the function.
#
# load_users():
# Reads user information from users.json.
#
# ROOT / "config" / "users.json":
# Builds the path to the user configuration file.
#
# str():
# Converts the path into a string.


@st.cache_data
def permission_config():
    return load_document_permissions(str(ROOT / "config" / "document_permissions.csv"))

# @st.cache_data:
# Caches the permission configuration.
#
# permission_config():
# A function used to load document permissions.
#
# load_document_permissions():
# Reads the allowed roles for each document.
#
# document_permissions.csv:
# Stores the document access rules.
#
# Example:
# A report may be accessible to Admin, Analyst and Viewer.


# ------------------------------------------------
# SECTION 6: INITIALIZING USER SESSION
# ------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

# if:
# Checks whether a condition is true.
#
# "user":
# A key used to store the currently logged-in user's information.
#
# not in:
# Checks whether the specified key does not exist.
#
# st.session_state:
# A Streamlit feature that maintains information across application reruns.
#
# user = None:
# Initially, no user is logged in.
#
# None:
# A Python special value representing the absence of a value.


# ------------------------------------------------
# SECTION 7: USER LOGIN PAGE
# ------------------------------------------------

if st.session_state.user is None:

    # Checks whether the current user is not logged in.
    # If no user exists, the login page is displayed.

    st.subheader("Sign in")
    # Displays a smaller heading on the login page.


    st.info("Local classroom prototype only. Change demo credentials before any shared deployment.")
    # Displays an informational message.
    # It reminds users that the current credentials are for demonstration.


    with st.form("login"):
        # with:
        # Used to execute a block of code inside a context manager.
        #
        # st.form():
        # Groups input fields into a single form.
        #
        # "login":
        # Unique name assigned to the login form.

        username = st.text_input("Username")
        # text_input():
        # Creates a text box where the user enters their username.
        #
        # username:
        # Stores the entered username.


        password = st.text_input("Password", type="password")
        # Creates the password input field.
        #
        # type="password":
        # Hides the entered password characters on the screen.
        #
        # password:
        # Stores the entered password.


        submitted = st.form_submit_button("Sign in")
        # Creates the Sign in button.
        #
        # form_submit_button():
        # Submits all the values inside the form.
        #
        # submitted:
        # Stores True if the user submits the form.
        # Otherwise, it is False.


    if submitted:

        # Executes the following code when the form is submitted.

        user = verify_password(username.strip(), password, users_config())

        # verify_password():
        # Checks the entered login credentials.
        #
        # username.strip():
        # Removes unnecessary spaces from the beginning and end
        # of the entered username.
        #
        # password:
        # The password entered by the user.
        #
        # users_config():
        # Loads the configured user details.
        #
        # user:
        # Stores the authenticated user's information if valid.


        if user:

            # Checks whether the login credentials are valid.

            st.session_state.user = user
            # Saves the authenticated user in Streamlit session state.
            # This allows the application to remember the login.

            st.rerun()
            # Reruns the complete Streamlit application.
            # The login page is replaced by the main application.


        else:

            # Executes when the login credentials are invalid.

            st.error("Invalid username or password.")
            # Displays an error message on the website.


    st.markdown("**Demo accounts:** `admin / admin123`, `analyst / analyst123`, `viewer / viewer123`")

    # markdown():
    # Displays formatted Markdown content.
    #
    # ** **:
    # Makes text bold.
    #
    # ` `:
    # Displays text in code formatting.
    #
    # Shows the demo login credentials.


    st.stop()
    # Stops the execution of the remaining application code.
    # The user must log in successfully before continuing.


# ------------------------------------------------
# SECTION 8: RETRIEVING LOGGED-IN USER INFORMATION
# ------------------------------------------------

user = st.session_state.user

# Gets the current user's information from session state.
# The user variable contains information such as username and role.


with st.sidebar:
    # sidebar:
    # Creates a separate sidebar on the left side of the website.
    #
    # All components inside this block appear in the sidebar.


    st.success(f"Signed in as **{user['username']}**")
    # success():
    # Displays a success-style message.
    #
    # f-string:
    # Allows Python variables to be inserted inside a string.
    #
    # user['username']:
    # Retrieves the logged-in user's username.


    st.write(f"Role: `{user['role']}`")
    # write():
    # Displays text or other information on the Streamlit page.
    #
    # user['role']:
    # Retrieves the role assigned to the logged-in user.


    if st.button("Sign out"):

        # button():
        # Creates a clickable button.
        #
        # If the user clicks Sign out, the following code executes.

        st.session_state.user = None
        # Clears the currently logged-in user.

        st.rerun()
        # Refreshes the application.
        # Since the user is now None, the login page appears again.


    st.divider()
    # divider():
    # Displays a horizontal line to separate sidebar sections.


    llm_name = st.text_input("Ollama model", value="qwen2.5:3b")

    # Allows the user to enter the language model name.
    #
    # value="qwen2.5:3b":
    # Sets the default model name.
    #
    # llm_name:
    # Stores the selected model name.
    #
    # Ollama uses this model to generate answers.


    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

    # slider():
    # Creates a slider control.
    #
    # min_value=1:
    # Minimum number of chunks that can be retrieved.
    #
    # max_value=8:
    # Maximum number of chunks that can be retrieved.
    #
    # value=4:
    # Default number of retrieved chunks.
    #
    # top_k:
    # Stores the number of relevant chunks to retrieve.


    st.caption("Permission filtering is applied before embeddings/retrieval for this user's session.")

    # Explains that document permissions are checked
    # before embedding and retrieval for the current user.


# ------------------------------------------------
# SECTION 9: SELECTING DOCUMENT COLLECTIONS
# ------------------------------------------------

folder = ROOT / "data" / "financial_reports"

# Creates the path to the official financial report folder.
# This folder contains the actual financial reports used in the project.


synthetic_folder = ROOT / "data" / "synthetic_smoke_test"

# Creates the path to the synthetic smoke-test folder.
# Synthetic documents are used for testing the application.


source_choice = st.radio("Document collection", ["Official report folder", "Synthetic smoke-test folder"], horizontal=True)

# radio():
# Creates radio-button options.
#
# The user can select one document collection.
#
# Official report folder:
# Contains actual financial reports.
#
# Synthetic smoke-test folder:
# Contains sample test documents.
#
# horizontal=True:
# Displays the radio buttons horizontally.
#
# source_choice:
# Stores the selected option.


selected_folder = folder if source_choice == "Official report folder" else synthetic_folder

# This is a conditional expression, also called a ternary expression.
#
# if source_choice == "Official report folder":
# Selects the official report folder if that option is chosen.
#
# else:
# Otherwise, selects the synthetic folder.
#
# selected_folder:
# Stores the final folder path selected by the user.


# ------------------------------------------------
# SECTION 10: LOADING AND FILTERING DOCUMENTS
# ------------------------------------------------

all_docs = load_documents(selected_folder)

# load_documents():
# Loads all supported documents from the selected folder.
#
# selected_folder:
# The folder chosen by the user.
#
# all_docs:
# Stores the documents discovered by the loading function.


permissions = permission_config()

# Loads the document permission rules from the CSV file.
#
# permissions:
# Stores the permission information.


permitted_docs = filter_documents_for_role(all_docs, user["role"], permissions)

# filter_documents_for_role():
# Checks which documents the logged-in user is allowed to access.
#
# all_docs:
# All documents found in the selected folder.
#
# user["role"]:
# The current user's role, such as Admin, Analyst or Viewer.
#
# permissions:
# The document permission rules.
#
# permitted_docs:
# Stores only the documents the user is permitted to access.


denied_count = len(all_docs) - len(permitted_docs)

# len():
# Returns the number of items in a collection.
#
# len(all_docs):
# Total number of discovered documents.
#
# len(permitted_docs):
# Number of documents allowed for the current user.
#
# denied_count:
# Number of documents filtered out by the access policy.


# ------------------------------------------------
# SECTION 11: DISPLAYING DOCUMENT STATISTICS
# ------------------------------------------------

col1, col2, col3 = st.columns(3)

# columns(3):
# Divides the Streamlit page into three equal columns.
#
# col1, col2 and col3:
# Variables that represent the three columns.


col1.metric("Documents found", len(all_docs))

# metric():
# Displays a large numerical value with a label.
#
# Shows the total number of documents found.


col2.metric("Documents permitted", len(permitted_docs))

# Displays the number of documents the current user can access.


col3.metric("Filtered by access policy", denied_count)

# Displays the number of documents blocked by role permissions.


# ------------------------------------------------
# SECTION 12: CHECKING DOCUMENT ACCESS
# ------------------------------------------------

if not permitted_docs:

    # Checks whether the permitted document list is empty.
    #
    # not:
    # Reverses a Boolean condition.
    #
    # If there are no accessible documents, this condition is True.

    st.warning("No permitted documents found. Add reports and register each filename in config/document_permissions.csv with allowed roles. Unknown files are denied to non-admin roles.")

    # warning():
    # Displays a warning message.
    #
    # It explains that documents need to be added and registered
    # in the permission configuration file.


    st.stop()
    # Stops further execution because no permitted documents exist.


# ------------------------------------------------
# SECTION 13: DOCUMENT CHUNKING AND EMBEDDING
# ------------------------------------------------

with st.spinner("Creating searchable embeddings for permitted documents..."):

    # spinner():
    # Displays a loading animation while the operations are running.
    #
    # This informs the user that document processing is in progress.


    rows = build_rows(permitted_docs)

    # build_rows():
    # Processes the permitted documents.
    #
    # It extracts and organizes document text into smaller chunks.
    #
    # The chunks are associated with metadata such as:
    # Source filename, page number and chunk ID.
    #
    # rows:
    # Stores the processed chunks and their metadata.


    model, rows, vectors = embed_rows(rows)

    # embed_rows():
    # Generates embeddings for the processed text chunks.
    #
    # model:
    # The sentence embedding model used by the application.
    #
    # rows:
    # The processed text chunks and associated metadata.
    #
    # vectors:
    # Numerical representations of the text chunks.
    #
    # Embeddings allow the system to compare the meaning
    # of the user's question with document content.


# ------------------------------------------------
# SECTION 14: DISPLAYING INDEXING INFORMATION
# ------------------------------------------------

st.caption(f"Indexed {len(permitted_docs)} permitted documents into {len(rows)} text chunks.")

# Displays the number of permitted documents and generated chunks.
#
# len(permitted_docs):
# Number of documents accessible to the current user.
#
# len(rows):
# Total number of text chunks created from those documents.


# ------------------------------------------------
# SECTION 15: TAKING USER QUESTIONS
# ------------------------------------------------

question = st.text_input("Ask a question about the selected reports", placeholder="What was the reported revenue for FY2025?")

# text_input():
# Creates a text box for entering a financial question.
#
# question:
# Stores the question entered by the user.
#
# placeholder:
# Shows an example question inside the empty input field.


colA, colB = st.columns([1, 1])

# Creates two columns of equal width.
# These columns are used to display the two action buttons.


with colA:

    run_agent = st.button("Ask FinSight Agent", type="primary")

    # Creates the main RAG button.
    #
    # type="primary":
    # Gives the button primary visual importance.
    #
    # run_agent:
    # Becomes True when the user clicks the button.


with colB:

    run_baseline = st.button("Run LLM-only baseline")

    # Creates the baseline comparison button.
    #
    # run_baseline:
    # Becomes True when the user clicks this button.
    #
    # The baseline generates an answer without document retrieval.


# ------------------------------------------------
# SECTION 16: RUNNING THE RAG-BASED AGENT
# ------------------------------------------------

if run_agent and question.strip():

    # run_agent:
    # Checks whether the Ask FinSight Agent button was clicked.
    #
    # and:
    # Logical operator that requires both conditions to be True.
    #
    # question.strip():
    # Checks that the entered question is not empty or whitespace.
    #
    # The RAG workflow executes only when the button is clicked
    # and a valid question has been entered.


    result = answer_with_agent(question.strip(), model, rows, vectors, user["role"], top_k=top_k, llm_name=llm_name)

    # answer_with_agent():
    # Calls the main RAG question-answering function.
    #
    # question.strip():
    # The cleaned user question.
    #
    # model:
    # Sentence Transformer embedding model.
    #
    # rows:
    # Document chunks and their metadata.
    #
    # vectors:
    # Embeddings generated from document chunks.
    #
    # user["role"]:
    # Role of the logged-in user.
    #
    # top_k=top_k:
    # Number of relevant chunks to retrieve.
    #
    # llm_name=llm_name:
    # Name of the Ollama language model.
    #
    # result:
    # Stores the output returned by the RAG function.
    #
    # The returned result contains information such as:
    # Answer, question type, latency, retrieved evidence,
    # error information and abstention status.


    # --------------------------------------------
    # DISPLAYING THE GENERATED ANSWER
    # --------------------------------------------

    st.subheader("Answer")

    # Displays a heading named Answer.


    st.write(result["answer"])

    # Displays the answer returned by the RAG function.
    #
    # result["answer"]:
    # Accesses the answer field from the result dictionary.


    st.caption(f"Question type: {result['question_type']} • Generation latency: {result['latency']:.2f}s • Abstained: {result['abstained']}")

    # Displays additional information about the answer.
    #
    # result["question_type"]:
    # Displays the detected question category.
    #
    # result["latency"]:
    # Displays the time taken to generate the response.
    #
    # :.2f:
    # Formats the latency number to two decimal places.
    #
    # s:
    # Represents seconds.
    #
    # result["abstained"]:
    # Indicates whether the system abstained from answering.


    if result["error"]:

        # Checks whether an error was returned by the RAG function.

        st.error(result["error"])

        # Displays the error message if one exists.


    # --------------------------------------------
    # DISPLAYING RETRIEVED EVIDENCE
    # --------------------------------------------

    with st.expander("Retrieved evidence"):

        # expander():
        # Creates an expandable and collapsible section.
        #
        # The user can click it to inspect the retrieved evidence.


        for item in result["retrieved"]:

            # for:
            # A Python loop used to repeat an operation for each item.
            #
            # item:
            # Represents one retrieved evidence item.
            #
            # result["retrieved"]:
            # Contains the chunks retrieved by the RAG system.


            st.markdown(f"**{item['source']} — page {item['page']} — chunk {item['chunk_id']}** (similarity {item['score']:.3f})")

            # Displays the evidence metadata.
            #
            # item['source']:
            # Source document filename.
            #
            # item['page']:
            # Page number where the text was found.
            #
            # item['chunk_id']:
            # Unique identifier of the retrieved chunk.
            #
            # item['score']:
            # Similarity score of the retrieved chunk.
            #
            # :.3f:
            # Formats the similarity score to three decimal places.
            #
            # ** **:
            # Makes the metadata bold in Markdown.


            st.write(item["text"])

            # Displays the actual retrieved text passage.
            # This allows users to inspect the supporting evidence.


    st.session_state["last_result"] = result

    # Saves the latest RAG result in Streamlit session state.
    # It can remain available across application reruns.


# ------------------------------------------------
# SECTION 17: RUNNING THE LLM-ONLY BASELINE
# ------------------------------------------------

if run_baseline and question.strip():

    # Checks whether the baseline button was clicked
    # and whether the user entered a non-empty question.


    answer, latency, error = generate_answer(question.strip(), [], llm_name=llm_name, use_rag=False)

    # generate_answer():
    # Calls the language model to generate a response.
    #
    # question.strip():
    # The cleaned user question.
    #
    # []:
    # An empty list is passed instead of retrieved document chunks.
    #
    # llm_name=llm_name:
    # Specifies the Ollama model to use.
    #
    # use_rag=False:
    # Explicitly disables the RAG retrieval mode.
    #
    # This means the model answers without annual-report evidence.
    #
    # answer:
    # Stores the generated baseline answer.
    #
    # latency:
    # Stores the time taken to generate the answer.
    #
    # error:
    # Stores any error returned by the function.


    st.subheader("LLM-only baseline")

    # Displays the heading for the baseline result.


    if error:

        # Checks whether the baseline function returned an error.

        st.error(error)

        # Displays the error if one exists.


    else:

        # Executes when no error is returned.

        st.write(answer)

        # Displays the answer generated by the LLM-only baseline.


        st.caption(f"Generation latency: {latency:.2f}s. This baseline has no annual-report retrieval.")

        # Displays the baseline generation time.
        #
        # :.2f:
        # Formats latency to two decimal places.
        #
        # Explains that the baseline does not retrieve
        # information from the annual report.


# ------------------------------------------------
# SECTION 18: FINAL DIVIDER AND DISCLAIMER
# ------------------------------------------------

st.divider()

# Displays a horizontal divider to separate the main application
# from the final disclaimer.


st.caption("For research/demo use only. Verify every figure against official reports. Not investment advice.")

# Displays the final disclaimer.
#
# It reminds users that:
# 1. The application is a research and demonstration prototype.
# 2. Financial figures should be verified against official reports.
# 3. The application does not provide investment advice.


# ================================================================
# END OF FINSIGHT RAG MAIN APPLICATION
# ================================================================
