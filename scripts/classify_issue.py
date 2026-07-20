"""
Agent 1 - Issue Classifier
Reads the issue title/body, asks Claude to classify it, and applies a matching label.
Triggered by: .github/workflows/issue-classifier.yml on issues [opened, edited]
"""
import json
import os

import anthropic
from github_utils import add_label, get_issue_labels, post_comment

ALLOWED_LABELS = ["feature", "bug", "refactor", "docs", "question", "chore"]

SYSTEM_PROMPT = f"""You are an issue triage assistant for a software project.
Classify the GitHub issue into exactly one of these categories: {', '.join(ALLOWED_LABELS)}.

Respond with ONLY valid JSON, no markdown fences, no extra text, in this exact shape:
{{"label": "one of the allowed categories", "confidence": 0.0-1.0, "reasoning": "one short sentence"}}
"""


def classify(title: str, body: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Title: {title}\n\nBody:\n{body or '(no description provided)'}"}
        ],
    )
    text = message.content[0].text.strip()
    return json.loads(text)


def main():
    issue_number = int(os.environ["ISSUE_NUMBER"])
    title = os.environ.get("ISSUE_TITLE", "")
    body = os.environ.get("ISSUE_BODY", "")

    existing_labels = get_issue_labels(issue_number)
    if any(label in ALLOWED_LABELS for label in existing_labels):
        print(f"Issue #{issue_number} already has a classification label {existing_labels}, skipping.")
        return

    try:
        result = classify(title, body)
    except (json.JSONDecodeError, anthropic.APIError) as e:
        print(f"Classification failed, skipping: {e}")
        return

    label = result.get("label")

    if label not in ALLOWED_LABELS:
        print(f"Model returned an unexpected label '{label}', skipping label application.")
        return

    add_label(issue_number, label)
    print(f"Applied label '{label}' (confidence {result.get('confidence')}) to issue #{issue_number}")

    # Leave a short, transparent comment so humans can see the agent's reasoning and correct it if wrong.
    comment = (
        f"🤖 **Issue Classifier Agent**\n\n"
        f"Classified this as **`{label}`** (confidence: {result.get('confidence', 'n/a')}).\n"
        f"> {result.get('reasoning', '')}\n\n"
        f"_If this is wrong, just change the label — the agent won't overwrite manual edits._"
    )
    post_comment(issue_number, comment)


if __name__ == "__main__":
    main()