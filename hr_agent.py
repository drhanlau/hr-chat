#!/usr/bin/env python3
"""
HR Dataset Terminal Agent
An interactive agent that answers questions about HR data using natural language.
"""

import sqlite3
import os
import sys
import re
from anthropic import Anthropic

DB_PATH = "HR_Dataset.sqlite3"


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"


def format_markdown_for_terminal(text: str) -> str:
    """Convert markdown formatting to terminal ANSI colors."""
    lines = text.split('\n')
    result = []
    in_code_block = False

    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            if in_code_block:
                result.append(f"{Colors.DIM}{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            else:
                result.append(f"{Colors.DIM}{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            continue

        if in_code_block:
            result.append(f"{Colors.CYAN}{line}{Colors.RESET}")
            continue

        # Handle headers (# Header)
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            header_text = header_match.group(2)
            result.append(f"{Colors.BOLD}{Colors.GREEN}{header_text}{Colors.RESET}")
            continue

        # Handle horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', line):
            result.append(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
            continue

        # Process inline formatting
        formatted_line = line

        # Bold **text** or __text__ -> Yellow
        formatted_line = re.sub(
            r'\*\*(.+?)\*\*|__(.+?)__',
            lambda m: f"{Colors.BOLD}{Colors.YELLOW}{m.group(1) or m.group(2)}{Colors.RESET}",
            formatted_line
        )

        # Italic *text* or _text_ -> Magenta (avoid matching ** or __)
        formatted_line = re.sub(
            r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)',
            lambda m: f"{Colors.MAGENTA}{m.group(1) or m.group(2)}{Colors.RESET}",
            formatted_line
        )

        # Inline code `code` -> Cyan
        formatted_line = re.sub(
            r'`([^`]+)`',
            lambda m: f"{Colors.CYAN}{m.group(1)}{Colors.RESET}",
            formatted_line
        )

        # Bullet points
        bullet_match = re.match(r'^(\s*)([-*+])\s+(.+)$', formatted_line)
        if bullet_match:
            indent = bullet_match.group(1)
            content = bullet_match.group(3)
            formatted_line = f"{indent}{Colors.BRIGHT_BLUE}•{Colors.RESET} {content}"

        # Numbered lists
        numbered_match = re.match(r'^(\s*)(\d+\.)\s+(.+)$', formatted_line)
        if numbered_match:
            indent = numbered_match.group(1)
            number = numbered_match.group(2)
            content = numbered_match.group(3)
            formatted_line = f"{indent}{Colors.BRIGHT_BLUE}{number}{Colors.RESET} {content}"

        result.append(formatted_line)

    return '\n'.join(result)


def get_api_key() -> str:
    """Get API key from environment or prompt user."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    print("ANTHROPIC_API_KEY environment variable not set.")
    print("You can set it with: export ANTHROPIC_API_KEY='your-key-here'")
    print()
    api_key = input("Enter your Anthropic API key: ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)
    return api_key

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
Keep responses brief but complete."""


class HRAgent:
    def __init__(self):
        api_key = get_api_key()
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.db_path = DB_PATH

        if not os.path.exists(self.db_path):
            print(f"Error: Database '{self.db_path}' not found.")
            print("Please ensure the HR dataset SQLite database exists.")
            exit(1)

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

        # Calculate column widths
        widths = [len(str(col)) for col in columns]
        for row in results:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        # Build table
        lines = []
        header = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
        lines.append(header)
        lines.append("-" * len(header))

        for row in results:
            line = " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
            lines.append(line)

        return "\n".join(lines)

    def ask(self, question: str) -> str:
        """Process a question and return an answer."""
        # Add user question to history
        self.conversation_history.append({"role": "user", "content": question})

        # Get SQL query from Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history
        )

        assistant_message = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Extract and execute SQL
        sql = self.extract_sql(assistant_message)

        if not sql:
            return assistant_message

        try:
            columns, results = self.execute_query(sql)
            formatted_results = self.format_results(columns, results)

            # Get analysis from Claude
            analysis_request = f"""Question: {question}

SQL Query executed:
```sql
{sql}
```

Results:
{formatted_results}

Please provide a brief, helpful answer based on these results."""

            analysis_response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=ANALYSIS_PROMPT,
                messages=[{"role": "user", "content": analysis_request}]
            )

            analysis = analysis_response.content[0].text

            # Build response with clear sections
            output = f"\n{analysis}\n\n"
            output += f"---\n\n"
            output += f"**Query:**\n```sql\n{sql}\n```\n\n"
            output += f"**Results:**\n```\n{formatted_results}\n```"

            return output

        except Exception as e:
            return f"**Error executing query:** {e}\n\n**Generated SQL:**\n```sql\n{sql}\n```"

    def run(self):
        """Run the interactive terminal agent."""
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}HR Dataset Agent{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print("Ask questions about employee data in natural language.")
        print(f"Type {Colors.YELLOW}'quit'{Colors.RESET} or {Colors.YELLOW}'exit'{Colors.RESET} to end the session.")
        print(f"Type {Colors.YELLOW}'clear'{Colors.RESET} to reset conversation history.")
        print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}")
        print()

        while True:
            try:
                question = input(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}You:{Colors.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
                break

            if not question:
                continue

            if question.lower() in ('quit', 'exit'):
                print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
                break

            if question.lower() == 'clear':
                self.conversation_history = []
                print(f"{Colors.GREEN}Conversation history cleared.{Colors.RESET}")
                continue

            print()
            response = self.ask(question)
            formatted_response = format_markdown_for_terminal(response)
            print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}Agent:{Colors.RESET} {formatted_response}")
            print()


def main():
    agent = HRAgent()
    agent.run()


if __name__ == "__main__":
    main()
