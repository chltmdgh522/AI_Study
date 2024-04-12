import os
import subprocess

# Kaggle config 디렉토리를 설정
os.environ['KAGGLE_CONFIG_DIR'] = 'Project2_Photo1'

# Kaggle datasets download 명령어 실행
result = subprocess.run(['kaggle', 'datasets', 'download', 'dogs-vs-cats'])

# subprocess.run() 함수가 반환하는 CompletedProcess 객체의 반환 코드를 출력
print("Completed with return code:", result.returncode)
