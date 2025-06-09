# MemoriaX v2

MemoriaX v2 is a prototype conversational agent that stores past messages and
their vector embeddings in a SQLite database.  It can recall similar memories to
provide context-aware responses.  The project includes utilities for embedding
text, querying similar memories with FAISS and storing session data.

## Setup
1. Create a virtual environment and activate it.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   The list includes packages such as `flask`, `transformers`, `sentence-transformers`,
   `spacy` and `faiss-cpu`.
3. (Optional) Download the spaCy language model used by the NLP utilities:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the application
Run the command below from the repository root to start an interactive
conversation:
```bash
python memoriax2/main.py
```
During the session your input, the bot's responses and detected emotions are
logged to `memory.db`.  Type `exit` to end the conversation and receive a
summary of the session.

For a web demo using Flask see `app.py`.  The tests in `memoriax2/tests` show
examples of the memory indexing and recall features.

