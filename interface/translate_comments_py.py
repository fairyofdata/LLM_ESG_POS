import os
import random
import time
from deep_translator import GoogleTranslator

#Functions that translate the comments of the .py file
def translate_comments_in_python_file(file_path, translation_counter):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    Translation using # Googletranslator
    translator = GoogleTranslator(source='ko', target='en')

    #Find the comment and translate
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):  # 주석인 라인만 처리
            original_comment = line.strip()
            try:
                # translation
                translated_comment = translator.translate(original_comment)

                # Print the contents before and after translation
                print(f"Python File Translation #{translation_counter}:")
                print(f"Original: {original_comment}")
                print(f"Translated: {translated_comment}")

                # Replace with the translated tin in the original code
                lines[i] = lines[i].replace(original_comment, translated_comment)

                # Increased translation counter
                translation_counter += 1

                # Apply delay time for each translation
                delay_time = random.uniform(1.5, 2.0) * random.uniform(0.9, 1.1)  # 1.5~2.0초에 난수 곱하기
                print(f"Delaying for {delay_time:.3f} seconds...\n")
                time.sleep(delay_time)  # 지연 시간 적용

            except Exception as e:
                # Printed the error message when translation failed
                print(f"Translation failed for comment: {original_comment}")
                print(f"Error: {e}")
                continue  # 번역 실패 시 넘어가서 계속 진행

    # Save the translated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    return translation_counter

# All .py file processing in the directory (including subdirectory)
def process_python_files_in_directory(directory_path):
    translation_counter = 1  # 번역 카운터 초기화
    for root, dirs, files in os.walk(directory_path):  # 하위 디렉토리까지 순회
        for filename in files:
            if filename.endswith('.py'):  # .py 파일만 처리
                file_path = os.path.join(root, filename)

                #Note output of new files
                print(f"\nStarting translation for Python file: {filename}")

                # File processing
                translation_counter = translate_comments_in_python_file(file_path, translation_counter)

                # Progress output after file processing
                print(f"Processed: {filename}")


# Examples of execution
directory_path = 'C:\\Users\\stard\\Documents\\GitHub\\LLM_ESG_POS'  # 전체 프로젝트 경로
process_python_files_in_directory(directory_path)
