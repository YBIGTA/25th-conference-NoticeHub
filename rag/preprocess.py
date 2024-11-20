# data_preprocessing.py

import glob
import pandas as pd
from dotenv import load_dotenv


class DataPreprocessor:
    def __init__(self):
        self.combined_df = pd.DataFrame()

    def load_api_keys(self):
        # API 키를 환경변수로 관리하기 위한 설정 파일
        load_dotenv()

    def load_csv_files(self):
        # 'noticehub/data' 폴더 내 모든 CSV 파일 로드
        csv_files = glob.glob('../data/*.csv')
        print("로드된 CSV 파일 목록:")
        for file in csv_files:
            print(file)

        for file in csv_files:
            try:
                # 파일 읽기
                df = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8')
                # 비어있지 않은 데이터프레임만 연결
                if not df.empty:
                    self.combined_df = pd.concat([self.combined_df, df], ignore_index=True)
                else:
                    print(f"비어 있는 파일: {file} - 무시됨")
            except pd.errors.EmptyDataError:
                print(f"비어 있는 파일: {file} - 무시됨")
            except pd.errors.ParserError as e:
                print(f"구문 오류 발생 파일: {file} - {e}")

    def combine_csv_files(self, output_file):
        # 모든 CSV 파일을 결합하여 하나의 CSV 파일로 저장
        self.load_csv_files()  # CSV 파일 로드
        if not self.combined_df.empty:
            self.combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"결합된 CSV 파일이 '{output_file}'로 저장되었습니다.")
        else:
            print("결합할 데이터가 없습니다.")


# 사용 예시
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.load_api_keys()
    preprocessor.combine_csv_files('../data/combined_notices.csv')