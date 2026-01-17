# HR Data Agent

A web-based chat interface for querying HR data using natural language. Ask questions about employee satisfaction, turnover, performance, and more - the AI agent translates your questions into SQL queries and provides insightful answers.

![HR Data Agent Screenshot](screenshot.png)

## Features

- Natural language interface for HR data analysis
- Automatic SQL query generation
- Clean, modern chat UI (similar to Claude/ChatGPT)
- Markdown rendering for formatted responses
- Shows SQL queries and raw results for transparency
- Conversation history support

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **AI Model**: GLM-4.7 (OpenAI-compatible API)
- **Database**: SQLite

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hr-chat.git
cd hr-chat
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
```

### 5. Run the application

```bash
python app.py
```

Open http://localhost:5001 in your browser.

## Database Schema

The application uses an SQLite database (`HR_Dataset.sqlite3`) with an `employees` table containing:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Employee name |
| satisfaction_level | REAL | 0.0 to 1.0 |
| last_evaluation | REAL | 0.0 to 1.0 |
| number_project | INTEGER | Number of projects |
| average_monthly_hours | INTEGER | Average hours/month |
| exp_in_company | INTEGER | Years of experience |
| work_accident | INTEGER | 0 = no, 1 = yes |
| left | INTEGER | 0 = stayed, 1 = left |
| promotion_last_5years | INTEGER | 0 = no, 1 = yes |
| role | TEXT | Department/role |
| salary | TEXT | low, medium, high |

## Example Questions

- "How many employees are there?"
- "What is the average satisfaction level by department?"
- "Which roles have the highest turnover rate?"
- "Compare satisfaction levels between employees who left and stayed"
- "Show employees with low satisfaction but high evaluation scores"

## License

MIT
