# text_utils.py
import json


def get_line_text(line):
    # strip the line
    line = line.strip()

    # check for jsonl
    if line.startswith("{"):  # JSONL format
        try:
            # load the json
            data = json.loads(line)

            # check for a text attribute
            if "text" in data:
                # return the text
                return data["text"]
            # check for content attribute
            elif "content" in data:
                # return content
                return data["content"]
            else:
                # no text or content
                raise ValueError(f"No 'text' or 'content' field found in JSONL: {line}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSONL line: {line}, error: {e}") from e
    else:
        # Plain text format, return the line
        return line
