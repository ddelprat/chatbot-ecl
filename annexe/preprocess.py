import fitz  # PyMuPDF
import re
import torch
import sys

# Function to preprocess and extract text into meaningful passages
def preprocess_document(doc):
    passages = []
    hyphenated_line = None  # To store and handle hyphenated words

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("blocks")  # Extract text blocks
        for block in text:
            block_text = block[4].strip()  # Extract block content
            block_text = block_text.replace("\n", " ")
            lines = re.split(r'(?<=[.?!])\s+', block_text)
            passage = ""
            for line in lines:
                line = line.strip()
                # Handle continuation of hyphenated words
                if hyphenated_line:
                    if line.startswith("-"):
                        hyphenated_line += line[1:]  # Remove leading hyphen
                    else:
                        hyphenated_line += line
                    line = hyphenated_line
                    hyphenated_line = None

                if line.endswith("-"):
                    hyphenated_line = line[:-1]  # Store the part before the hyphen
                    continue  # Wait for the next line to complete the word

                if len(line) == 0:
                    continue

                # Start a new passage if the line is a bullet point or exceeds length
                if len(passage) + len(line) > 500 or re.match(r"^[-•—]", line):
                    if passage:
                        passages.append(passage.strip())  # Save the current passage
                    passage = line  # Start a new passage
                else:
                    passage += f" {line}"

            if passage:  # Save any remaining text as a passage
                if len(passage) > 50:
                    passages.append(passage.strip().replace("- ", ""))

    passages = [item for item in passages if len(item) > 40 and not (
                item.startswith("Article ") or item == "RÈGLEMENT DE SCOLARITÉ DE L’ECOLE CENTRALE DE LYON 2024-2025")]
    return passages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path/to/corpus/file>")
        sys.exit(1)
    file_path = sys.argv[1]
    doc = fitz.open(file_path)
    processed_passages = preprocess_document(doc)
    output_file = "preprocessed"+file_path[:-4]+".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed_passages:
            f.write(item + "\n")
    print(f"File '{output_file}' has been created successfully.")