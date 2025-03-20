import os
import nbformat
import random
import time
from deep_translator import GoogleTranslator


#Jupyter Notebook Functions translating the markdown of the file
def translate_markdown_in_notebook(notebook_path, translation_counter):
    # Read notbook file
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    Translation using # Googletranslator
    translator = GoogleTranslator(source='ko', target='en')

    # Translated Markdown by traveling around each cell
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'markdown':  # 마크다운 셀만 처리
            source_code = cell['source']
            # Translation of Hangul Text in Mark Down
            try:
                # translation
                translated_text = translator.translate(source_code)

                # Print the contents before and after translation
                print(f"Markdown Translation #{translation_counter}:")
                print(f"Original: {source_code}")
                print(f"Translated: {translated_text}")

                # Replace with the markdown translated from the original code
                source_code = translated_text

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
                print(f"Translation failed for markdown: {source_code}")
                print(f"Error: {e}")
                continue  # 번역 실패 시 넘어가서 계속 진행

    #Save the Notebook file to the changed content
    with open(notebook_path, 'w', encoding='utf-8') as notebook_file:
        nbformat.write(notebook_content, notebook_file)

    return translation_counter


# All .IPynb file processing in the directory (including subdirectory)
def process_notebooks_in_directory(directory_path):
    translation_counter = 1  # 번역 카운터 초기화
    for root, dirs, files in os.walk(directory_path):  # 하위 디렉토리까지 순회
        for filename in files:
            if filename.endswith('.ipynb'):  # .ipynb 파일만 처리
                file_path = os.path.join(root, filename)

                #Note output of new files
                print(f"\nStarting translation for markdown in file: {filename}")

                # File processing
                translation_counter = translate_markdown_in_notebook(file_path, translation_counter)

                # Latitude time and progress output
                print(f"Processed: {filename}")


# Examples of execution
directory_path = 'C:\\Users\\stard\\Documents\\GitHub\\LLM_ESG_POS'  # 전체 프로젝트 경로
process_notebooks_in_directory(directory_path)
