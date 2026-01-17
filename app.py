#!/usr/bin/env python3
"""
HR Dataset Web Agent
A web-based chat interface for querying HR data using natural language.
"""

import sqlite3
import os
from flask import Flask, render_template, request, jsonify
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

SYSTEM_PROMPT = f"""You are an HR data analyst assistant. You help answer questions about employee data by generating SQL queries.

Database Schema:
{SCHEMA}

When the user asks a question:
1. Generate a valid SQLite SQL query to answer their question
2. Return ONLY the SQL query wrapped in ```sql``` code blocks
3. Keep queries efficient and use appropriate aggregations
4. For percentage calculations, multiply by 100 and round to 2 decimal places
5. Limit results to 20 rows unless the user asks for more
6. Use descriptive column aliases for readability

If the question cannot be answered with the available data, explain why.
If the question is ambiguous, make reasonable assumptions and state them.
"""

ANALYSIS_PROMPT = """Based on the user's question and the query results, provide a clear, concise answer.
Be conversational but informative. Include specific numbers from the results.
If the results are empty, explain what that means in context.
Keep responses brief but complete. Use markdown formatting for readability."""


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

    def ask(self, question: str) -> dict:
        """Process a question and return an answer with metadata."""
        try:
            self.conversation_history.append({"role": "user", "content": question})

            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=messages
            )

            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            sql = self.extract_sql(assistant_message)

            if not sql:
                return {
                    "answer": assistant_message,
                    "sql": None,
                    "results": None
                }

            columns, results = self.execute_query(sql)
            formatted_results = self.format_results(columns, results)

            analysis_request = f"""Question: {question}

SQL Query executed:
```sql
{sql}
```

Results:
{formatted_results}

Please provide a brief, helpful answer based on these results."""

            analysis_response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": analysis_request}
                ]
            )

            analysis = analysis_response.choices[0].message.content

            return {
                "answer": analysis,
                "sql": sql,
                "results": formatted_results
            }

        except Exception as e:
            # Remove the failed message from history
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return {
                "answer": None,
                "sql": None,
                "results": None,
                "error": str(e)
            }

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


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "No message provided"}), 400

    try:
        hr_agent = get_agent()
        result = hr_agent.ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    global agent
    if agent:
        agent.clear_history()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
