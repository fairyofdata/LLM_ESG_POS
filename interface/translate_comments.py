import os
import nbformat
import random
import time
from deep_translator import GoogleTranslator

# File to track the already translated file
translated_files = set()


# Jupyter Notebook Functions translating
def translate_comments_in_notebook(notebook_path, translation_counter):
    # Read notbook file
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    Translation using # Googletranslator
    translator = GoogleTranslator(source='ko', target='en')

    # Translate tin by turning around each cell
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            source_code = cell['source']
            comments = [line for line in source_code.split('\n') if line.strip().startswith('#')]
            for comment in comments:
                try:
                    # Check that the comment does not contain the code (for example, excluding code parts such as 'data = {')
                    if '=' in comment or '{' in comment:  # 코드 일부가 포함된 주석은 번역하지 않음
                        print(f"Skipping comment as it contains code: {comment}")
                        continue

                    # translation
                    translated_comment = translator.translate(comment)

                    # Print the contents before and after translation
                    print(f"Translation #{translation_counter}:")
                    print(f"Original: {comment}")
                    print(f"Translated: {translated_comment}")

                    # Replace with the translated tin in the original code
                    source_code = source_code.replace(comment, translated_comment)

                    # Increased translation counter
                    translation_counter += 1

                    # Application of the content after translation
                    cell['source'] = source_code

                    # Apply delay time for each translation
                    delay_time = random.uniform(1.5, 2.0) * random.uniform(0.9, 1.1)  # 1.5~2.0초에 난수 곱하기
                    print(f"Delaying for {delay_time:.3f} seconds...\n")
                    time.sleep(delay_time)  # 지연 시간 적용

                except Exception as e:
                    # Printed the error message when translation failed
                    print(f"Translation failed for comment: {comment}")
                    print(f"Error: {e}")
                    continue  # 번역 실패 시 넘어가서 계속 진행

    #Save the Notebook file to the changed content
    with open(notebook_path, 'w', encoding='utf-8') as notebook_file:
        nbformat.write(notebook_content, notebook_file)

    return translation_counter


# All .IPynb file processing in the directory (including subdirectory)
def process_notebooks_in_directory(directory_path,
                                   start_file="240830 Generalized Adequate Text Labeler Modeling.ipynb"):
    translation_counter = 1  # 번역 카운터 초기화
    start_processing = False  # 번역을 시작할 지 여부

    for root, dirs, files in os.walk(directory_path):  # 하위 디렉토리까지 순회
        for filename in files:
            if filename.endswith('.ipynb'):  # .ipynb 파일만 처리
                file_path = os.path.join(root, filename)

                # Skip the previously translated files
                if filename in translated_files:
                    print(f"Skipping already translated file: {filename}")
                    continue

                # Set translation starting point based on the file name
                if filename == start_file or start_processing:
                    print(f"\nStarting translation for file: {filename}")
                    start_processing = True

                    # File processing
                    translation_counter = translate_comments_in_notebook(file_path, translation_counter)

                    # Add to translated_files after file processing
                    translated_files.add(filename)

                # Latitude time and progress output
                print(f"Processed: {filename}")


# Examples of execution
directory_path = 'C:\\Users\\stard\\Documents\\GitHub\\LLM_ESG_POS'  # 전체 프로젝트 경로
process_notebooks_in_directory(directory_path)
