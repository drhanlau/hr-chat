#!/usr/bin/env python3
"""
HR Dataset Web Agent
A web-based chat interface for querying HR data using natural language.
"""

import sqlite3
import os
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DB_PATH = "HR_Dataset.sqlite3"

SCHEMA = """
Table: employees
Columns:
  - id: INTEGER (primary key, auto-increment)
  - name: TEXT (employee name)
  - satisfaction_level: REAL (0.0 to 1.0)
  - last_evaluation: REAL (0.0 to 1.0)
  - number_project: INTEGER (number of projects worked on)
  - average_monthly_hours: INTEGER (average hours worked per month)
  - exp_in_company: INTEGER (years of experience in company)
  - work_accident: INTEGER (0 = no accident, 1 = had accident)
  - left: INTEGER (0 = still employed, 1 = left company)
  - promotion_last_5years: INTEGER (0 = no promotion, 1 = promoted)
  - role: TEXT (department/role: sales, technical, support, management, IT, product_mng, marketing, RandD, accounting, hr)
  - salary: TEXT (low, medium, high)
"""

SYSTEM_PROMPT = f"""You are an expert HR assistant with two capabilities:

1. **Data Analysis**: You can query an employee database to answer data-driven questions
2. **General HR Knowledge**: You can answer general HR questions about best practices, policies, strategies, and advice

## Database Schema (for data queries):
{SCHEMA}

## How to respond:

**For DATA questions** (requiring database lookup):
- Questions about specific employee metrics, counts, averages, comparisons, trends
- Examples: "How many employees left?", "What's the average satisfaction?", "Which department has highest turnover?"
- Response: Generate a valid SQLite SQL query wrapped in ```sql``` code blocks
- Keep queries efficient, limit to 20 rows unless specified
- Use descriptive column aliases

**For GENERAL HR questions** (not requiring data):
- Questions about HR best practices, policies, strategies, career advice, management tips
- Examples: "How to improve employee retention?", "What causes burnout?", "How to conduct performance reviews?"
- Response: Provide a helpful, informative answer directly WITHOUT any SQL code
- Use markdown formatting for readability
- Draw from HR best practices and industry knowledge

**Decision Rule**: Only generate SQL if the question specifically requires data from the employee database. For conceptual, strategic, or advice-based questions, answer directly.
"""

ANALYSIS_PROMPT = """Based on the user's question and the query results, provide a clear, concise answer.
Be conversational but informative. Include specific numbers from the results.
If the results are empty, explain what that means in context.
Keep responses brief but complete. Use markdown formatting for readability.
When appropriate, add brief insights or recommendations based on the data."""


class HRAgent:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "GLM-4.7"
        self.conversation_history = []
        self.db_path = DB_PATH

    def execute_query(self, sql: str) -> tuple[list, list]:
        """Execute SQL query and return results with column names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description] if cursor.description else []
            results = cursor.fetchall()
            return columns, results
        except sqlite3.Error as e:
            raise Exception(f"SQL Error: {e}")
        finally:
            conn.close()

    def extract_sql(self, response: str) -> str | None:
        """Extract SQL query from response."""
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        return None

    def format_results(self, columns: list, results: list) -> str:
        """Format query results as a readable table."""
        if not results:
            return "No results found."

        widths = [len(str(col)) for col in columns]
        for row in results:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        lines = []
        header = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
        lines.append(header)
        lines.append("-" * len(header))

        for row in results:
            line = " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
            lines.append(line)

        return "\n".join(lines)

    def ask_stream(self, question: str):
        """Process a question and yield streaming responses."""
        try:
            self.conversation_history.append({"role": "user", "content": question})

            # Phase 1: Generate initial response (may contain SQL or direct answer)
            yield json.dumps({"type": "status", "content": "Thinking..."}) + "\n"

            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

            # Stream the first response
            full_response = ""
            yield json.dumps({"type": "phase", "content": "generating"}) + "\n"

            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield json.dumps({"type": "token", "content": content}) + "\n"

            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Check if response contains SQL
            sql = self.extract_sql(full_response)

            if not sql:
                # No SQL - this is a general HR answer
                yield json.dumps({
                    "type": "done",
                    "sql": None,
                    "results": None
                }) + "\n"
                return

            # Phase 2: Execute SQL query
            yield json.dumps({"type": "status", "content": "Executing SQL query..."}) + "\n"
            yield json.dumps({"type": "sql", "content": sql}) + "\n"

            try:
                columns, results = self.execute_query(sql)
                formatted_results = self.format_results(columns, results)
                yield json.dumps({"type": "results", "content": formatted_results}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "content": f"SQL Error: {e}"}) + "\n"
                return

            # Phase 3: Analyze results with streaming
            yield json.dumps({"type": "status", "content": "Analyzing results..."}) + "\n"
            yield json.dumps({"type": "phase", "content": "analyzing"}) + "\n"

            analysis_request = f"""Question: {question}

SQL Query executed:
```sql
{sql}
```

Results:
{formatted_results}

Please provide a brief, helpful answer based on these results."""

            analysis_stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": analysis_request}
                ],
                stream=True
            )

            for chunk in analysis_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield json.dumps({"type": "analysis_token", "content": content}) + "\n"

            yield json.dumps({
                "type": "done",
                "sql": sql,
                "results": formatted_results
            }) + "\n"

        except Exception as e:
            # Remove the failed message from history
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# Global agent instance (per session in production, use session management)
agent = None


def get_agent():
    global agent
    if agent is None:
        agent = HRAgent()
    return agent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "No message provided"}), 400

    def generate():
        hr_agent = get_agent()
        for chunk in hr_agent.ask_stream(question):
            yield f"data: {chunk}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/clear", methods=["POST"])
def clear():
    global agent
    if agent:
        agent.clear_history()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
