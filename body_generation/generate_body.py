"""
LLM_E: 실행기 LLM
유저의 취향 파악 후 후보 뉴스의 기대 본문 생성
"""

import os
import yaml
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Optional
import json
from pathlib import Path


class BodyGenerator:
    def __init__(self, 
                 prompt_path: str = "body_generation/prompt.yaml",
                 settings_path: str = "body_generation/generation_settings.yaml",
                 news_data_path: str = "dataset/MIND/MIND_news.tsv",
                 train_data_path: str = "dataset/MIND/MIND_train_(1000).tsv",
                 test_data_path: str = "dataset/MIND/MIND_test_(1000).tsv",
                 use_test: bool = False,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        """
        Args:
            prompt_path: 프롬프트 YAML 파일 경로
            settings_path: 생성 설정 YAML 파일 경로
            news_data_path: 뉴스 데이터 TSV 파일 경로
            train_data_path: 학습 데이터 TSV 파일 경로 (유저 클릭 히스토리)
            test_data_path: 테스트 데이터 TSV 파일 경로 (유저 클릭 히스토리)
            use_test: True면 test 데이터 사용, False면 train 데이터 사용
            api_key: OpenAI API 키 (없으면 환경변수 OPENAI_API_KEY 사용)
            model: 사용할 모델명
        """
        self.prompt_path = prompt_path
        self.settings_path = settings_path
        self.news_data_path = news_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.use_test = use_test
        self.model = model
        
        # API 키 설정
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공하세요.")
        self.client = OpenAI(api_key=api_key)
        
        # 데이터 로딩
        self._load_data()
        
        # 설정 로딩
        self._load_settings()
        
        # 프롬프트 로딩
        self._load_prompt()
    
    def _load_data(self):
        """뉴스 데이터와 학습/테스트 데이터 로딩"""
        data_type = "테스트" if self.use_test else "학습"
        print(f"데이터 로딩 중... ({data_type} 데이터 사용)")
        
        # 뉴스 데이터: news_id, category, subcategory, title, body
        self.news_df = pd.read_csv(
            self.news_data_path, 
            sep='\t', 
            names=['news_id', 'category', 'subcategory', 'title', 'body']
        )
        # news_dict를 더 간단하게 생성
        self.news_dict = {}
        for _, row in self.news_df.iterrows():
            self.news_dict[row['news_id']] = {
                'title': row['title'],
                'category': row['category'],
                'subcategory': row['subcategory']
            }
        
        # 학습/테스트 데이터: user, clicked_news, candidate_news, clicked
        data_path = self.test_data_path if self.use_test else self.train_data_path
        self.train_df = pd.read_csv(
            data_path,
            sep='\t',
            names=['user', 'clicked_news', 'candidate_news', 'clicked']
        )
        # user 컬럼을 int로 변환
        self.train_df['user'] = pd.to_numeric(self.train_df['user'], errors='coerce').astype('Int64')
        # NaN 값 제거
        self.train_df = self.train_df.dropna(subset=['user', 'clicked_news'])
        
        print(f"뉴스 데이터: {len(self.news_df)}개")
        print(f"{data_type} 데이터: {len(self.train_df)}개")
        print(f"뉴스 딕셔너리: {len(self.news_dict)}개")
    
    def _load_settings(self):
        """생성 설정 YAML 파일 로딩 및 파싱"""
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 설정을 딕셔너리로 파싱
        self.settings_dict = {}
        current_category = None
        current_key = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 카테고리 감지 (예: [Tone], [Abstraction Level])
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1].strip()
                self.settings_dict[current_category] = {}
                current_key = None
            # 설정값 감지 (예: {neutral}, {mixed})
            elif line.startswith('{') and line.endswith('}'):
                current_key = line[1:-1].strip()
                if current_category:
                    self.settings_dict[current_category][current_key] = ""
            # 설명 텍스트 (설정값 다음 줄)
            elif current_category and current_key and line:
                if self.settings_dict[current_category][current_key]:
                    self.settings_dict[current_category][current_key] += " " + line
                else:
                    self.settings_dict[current_category][current_key] = line
    
    def _load_prompt(self):
        """프롬프트 YAML 파일 로딩"""
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def _get_user_click_history(self, user_id: int, max_items: int = 10) -> List[str]:
        """
        유저의 클릭 히스토리에서 최근 N개 뉴스의 제목만 반환
        
        Args:
            user_id: 유저 ID
            max_items: 최대 개수 (기본값 10), 10개 이상이면 최근 10개, 적으면 전부 사용
        
        Returns:
            뉴스 제목 리스트 ['제목1', '제목2', ...]
        """
        # 해당 유저의 데이터 필터링
        user_data = self.train_df[self.train_df['user'] == user_id]
        
        if len(user_data) == 0:
            return []
        
        # clicked_news 컬럼에서 뉴스 ID들 파싱 (첫 번째 행만 사용, 모든 행이 동일함)
        clicked_news_str = str(user_data.iloc[0]['clicked_news'])
        clicked_news_list = clicked_news_str.split()
        
        # 10개 이상이면 최근 10개, 적으면 전부 사용
        if len(clicked_news_list) > max_items:
            recent_news_ids = clicked_news_list[-max_items:]
        else:
            recent_news_ids = clicked_news_list
        
        # 뉴스 제목만 반환
        result = []
        missing_count = 0
        missing_ids = []
        for news_id in recent_news_ids:
            if news_id in self.news_dict:
                news_info = self.news_dict[news_id]
                result.append(news_info['title'])
            else:
                missing_count += 1
                missing_ids.append(news_id)
        
        # 일부 뉴스가 없어도 경고만 출력하고 계속 진행
        if missing_count > 0:
            print(f"경고: {missing_count}개의 뉴스 ID를 찾을 수 없습니다. (총 {len(recent_news_ids)}개 중)")
            if missing_count <= 5:  # 5개 이하면 샘플 출력
                print(f"  찾을 수 없는 ID 샘플: {missing_ids[:5]}")
        
        if len(result) == 0 and len(recent_news_ids) > 0:
            print(f"오류: 모든 뉴스 ID를 찾을 수 없습니다. 첫 번째 ID: {recent_news_ids[0]}")
            print(f"  news_dict에 있는 샘플 키: {list(self.news_dict.keys())[:5]}")
        
        return result
    
    def _build_prompt(self, 
                     user_history: List[str], 
                     candidate_title: str) -> str:
        """
        프롬프트 생성
        
        Args:
            user_history: 유저 클릭 히스토리 (제목 리스트)
            candidate_title: 후보 뉴스 제목
        """
        prompt = self.prompt_template
        
        # 유저 히스토리 채우기 (최대 10개)
        for i in range(1, 11):
            if i <= len(user_history):
                news_str = user_history[i-1]  # 제목만 사용
            else:
                news_str = ""  # 빈 문자열로 채움
            prompt = prompt.replace(f"{{news{i}}}", news_str)
        
        # 후보 뉴스 제목 채우기 (제목만 사용)
        prompt = prompt.replace("{candidate_news}", candidate_title)
        
        # 설정값들을 generation_settings.yaml의 설명으로 교체
        # 각 카테고리별로 설정값 교체
        setting_mappings = {
            "Tone": "neutral",
            "Abstraction Level": "mixed",
            "Speculation Count": "1",
            "Length Bucket": "medium",
            "Format": "narrative"
        }
        
        for category, setting_key in setting_mappings.items():
            # {설정값} 패턴 찾기
            pattern = f"{category}: {{{setting_key}}}"
            if pattern in prompt:
                description = self.settings_dict.get(category, {}).get(setting_key, "")
                if description:
                    prompt = prompt.replace(pattern, f"{category}: {description}")
        
        return prompt
    
    def generate_body(self, 
                     user_id: int, 
                     candidate_news_id: str,
                     save_path: Optional[str] = None) -> Dict:
        """
        기대 본문 생성
        
        Args:
            user_id: 유저 ID
            candidate_news_id: 후보 뉴스 ID
            save_path: 결과 저장 경로 (선택)
        
        Returns:
            생성 결과 딕셔너리
        """
        # 유저 클릭 히스토리 가져오기 (10개 이상이면 최근 10개, 적으면 전부)
        user_history = self._get_user_click_history(user_id, max_items=10)
        
        if len(user_history) == 0:
            raise ValueError(f"유저 {user_id}의 클릭 히스토리가 없습니다.")
        
        # 후보 뉴스 정보 가져오기
        if candidate_news_id not in self.news_dict:
            raise ValueError(f"뉴스 ID {candidate_news_id}를 찾을 수 없습니다.")
        
        candidate_news = self.news_dict[candidate_news_id]
        candidate_title = candidate_news['title']
        
        # 프롬프트 생성
        prompt = self._build_prompt(
            user_history=user_history,
            candidate_title=candidate_title
        )
        
        # ChatGPT API 호출
        print(f"\n유저 {user_id}의 후보 뉴스 '{candidate_title}'에 대한 기대 본문 생성 중...")
        print(f"유저 히스토리: {len(user_history)}개 뉴스 사용")
        print("\n=== 전달된 프롬프트 ===")
        print(prompt)
        print("=" * 50)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            generated_body = response.choices[0].message.content.strip()
            
            result = {
                'user_id': user_id,
                'candidate_news_id': candidate_news_id,
                'candidate_title': candidate_title,
                'user_history_count': len(user_history),
                'user_history': user_history,
                'prompt': prompt,
                'generated_body': generated_body,
                'model': self.model
            }
            
            # 결과 저장
            if save_path:
                # train/test 폴더 구분
                dataset_type = "test" if self.use_test else "train"
                # save_path에서 base_dir 추출 (output 디렉토리)
                if os.path.dirname(save_path):
                    base_dir = os.path.dirname(save_path)
                else:
                    # 파일명만 있는 경우 현재 디렉토리 사용
                    base_dir = "."
                # output/train 또는 output/test 폴더 생성
                dataset_dir = os.path.join(base_dir, dataset_type)
                # user_id 기반 폴더 생성
                user_dir = os.path.join(dataset_dir, f"user_{user_id}")
                os.makedirs(user_dir, exist_ok=True)
                # 파일명만 사용하여 user 폴더 안에 저장
                filename = os.path.basename(save_path)
                final_path = os.path.join(user_dir, filename)
                self._save_result(result, final_path)
            
            print("생성 완료!")
            return result
            
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
            raise
    
    def generate_bodies_for_user(self,
                                 user_id: int,
                                 output_dir: Optional[str] = None) -> List[Dict]:
        """
        특정 유저의 모든 candidate_news에 대해 기대 본문 생성
        
        Args:
            user_id: 유저 ID
            output_dir: 결과 저장 디렉토리 (선택, 없으면 저장하지 않음)
        
        Returns:
            생성 결과 리스트
        """
        # 해당 유저의 데이터 필터링
        user_data = self.train_df[self.train_df['user'] == user_id]
        
        if len(user_data) == 0:
            raise ValueError(f"유저 {user_id}의 데이터가 없습니다.")
        
        # 유저 클릭 히스토리 가져오기 (한 번만)
        user_history = self._get_user_click_history(user_id, max_items=10)
        
        if len(user_history) == 0:
            raise ValueError(f"유저 {user_id}의 클릭 히스토리가 없습니다.")
        
        # 모든 행의 candidate_news 수집
        all_candidate_news_ids = []
        for _, row in user_data.iterrows():
            candidate_news_str = str(row['candidate_news'])
            candidate_news_ids = candidate_news_str.split()
            all_candidate_news_ids.extend(candidate_news_ids)
        
        # 중복 제거
        unique_candidate_news_ids = list(dict.fromkeys(all_candidate_news_ids))  # 순서 유지하면서 중복 제거
        
        print(f"\n유저 {user_id}에 대해 {len(unique_candidate_news_ids)}개의 후보 뉴스에 대한 본문을 생성합니다...")
        
        results = []
        for idx, candidate_news_id in enumerate(unique_candidate_news_ids, 1):
            if candidate_news_id not in self.news_dict:
                print(f"경고: 뉴스 ID {candidate_news_id}를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            candidate_title = self.news_dict[candidate_news_id]['title']
            
            # 프롬프트 생성
            prompt = self._build_prompt(
                user_history=user_history,
                candidate_title=candidate_title
            )
            
            # ChatGPT API 호출
            print(f"\n[{idx}/{len(unique_candidate_news_ids)}] 후보 뉴스 '{candidate_title}' 처리 중...")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                generated_body = response.choices[0].message.content.strip()
                
                result = {
                    'user_id': user_id,
                    'candidate_news_id': candidate_news_id,
                    'candidate_title': candidate_title,
                    'user_history_count': len(user_history),
                    'user_history': user_history,
                    'prompt': prompt,
                    'generated_body': generated_body,
                    'model': self.model
                }
                
                results.append(result)
                
                # 개별 파일로 저장
                if output_dir:
                    # train/test 폴더 구분
                    dataset_type = "test" if self.use_test else "train"
                    dataset_dir = os.path.join(output_dir, dataset_type)
                    # user_id 기반 폴더 생성
                    user_dir = os.path.join(dataset_dir, f"user_{user_id}")
                    os.makedirs(user_dir, exist_ok=True)
                    save_path = os.path.join(user_dir, f"news_{candidate_news_id}.json")
                    self._save_result(result, save_path)
                
                print("완료!")
                
            except Exception as e:
                print(f"오류 발생: {e}")
                continue
        
        # 전체 결과를 하나의 파일로도 저장
        if output_dir and results:
            # train/test 폴더 구분
            dataset_type = "test" if self.use_test else "train"
            dataset_dir = os.path.join(output_dir, dataset_type)
            # user_id 기반 폴더 생성
            user_dir = os.path.join(dataset_dir, f"user_{user_id}")
            os.makedirs(user_dir, exist_ok=True)
            all_results_path = os.path.join(user_dir, "all_results.json")
            with open(all_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n전체 결과가 {all_results_path}에 저장되었습니다.")
        
        print(f"\n총 {len(results)}개의 본문이 생성되었습니다.")
        return results
    
    def _save_result(self, result: Dict, save_path: str):
        """결과를 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {save_path}에 저장되었습니다.")


def main():
    """예제 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='뉴스 기대 본문 생성기')
    parser.add_argument('--user_id', type=int, default=None, help='유저 ID (지정하지 않으면 모든 유저 처리)')
    parser.add_argument('--start_user_id', type=int, default=None, help='시작 유저 ID (지정하면 해당 ID부터 이후 모든 유저 처리)')
    parser.add_argument('--candidate_news_id', type=str, default=None, help='후보 뉴스 ID (단일 뉴스 처리용, 없으면 모든 candidate_news 처리)')
    parser.add_argument('--output', type=str, default='body_generation/output', help='출력 디렉토리 경로')
    parser.add_argument('--use_test', action='store_true', help='테스트 데이터 사용 (기본값: 학습 데이터 사용)')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API 키 (선택, 환경변수 사용 가능)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='사용할 모델명')
    
    args = parser.parse_args()
    
    # 생성기 초기화
    generator = BodyGenerator(api_key=args.api_key, model=args.model, use_test=args.use_test)
    
    if args.candidate_news_id:
        # 단일 뉴스 처리 (user_id 필수)
        if args.user_id is None:
            raise ValueError("단일 뉴스 처리를 위해서는 --user_id를 지정해야 합니다.")
        # user 폴더 안에 저장하므로 파일명만 지정
        save_path = os.path.join(args.output, f"news_{args.candidate_news_id}.json")
        result = generator.generate_body(
            user_id=args.user_id,
            candidate_news_id=args.candidate_news_id,
            save_path=save_path
        )
        
        # 결과 출력
        print("\n=== 생성된 기대 본문 ===")
        print(result['generated_body'])
        print("\n=== 결과 요약 ===")
        print(f"유저 ID: {result['user_id']}")
        print(f"후보 뉴스: {result['candidate_title']}")
        print(f"사용된 히스토리: {result['user_history_count']}개")
    elif args.user_id is not None:
        # 특정 유저의 모든 candidate_news 처리
        results = generator.generate_bodies_for_user(
            user_id=args.user_id,
            output_dir=args.output
        )
        
        print(f"\n=== 생성 완료 ===")
        print(f"총 {len(results)}개의 본문이 생성되었습니다.")
    else:
        # 모든 유저 처리 (또는 start_user_id부터)
        all_user_ids = sorted(generator.train_df['user'].unique().tolist())
        
        # start_user_id가 지정된 경우 필터링
        if args.start_user_id is not None:
            all_user_ids = [uid for uid in all_user_ids if uid >= args.start_user_id]
            if len(all_user_ids) == 0:
                print(f"경고: 유저 ID {args.start_user_id} 이상인 유저가 없습니다.")
                return
            print(f"\n유저 ID {args.start_user_id}부터 이후 모든 유저에 대해 본문을 생성합니다. 총 {len(all_user_ids)}명의 유저...")
        else:
            print(f"\n모든 유저에 대해 본문을 생성합니다. 총 {len(all_user_ids)}명의 유저...")
        
        total_results = 0
        for user_idx, user_id in enumerate(all_user_ids, 1):
            print(f"\n{'='*60}")
            print(f"[{user_idx}/{len(all_user_ids)}] 유저 {user_id} 처리 중...")
            print(f"{'='*60}")
            
            try:
                results = generator.generate_bodies_for_user(
                    user_id=user_id,
                    output_dir=args.output
                )
                total_results += len(results)
                print(f"유저 {user_id}: {len(results)}개의 본문 생성 완료")
            except Exception as e:
                print(f"유저 {user_id} 처리 중 오류 발생: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"=== 전체 생성 완료 ===")
        print(f"총 {len(all_user_ids)}명의 유저, {total_results}개의 본문이 생성되었습니다.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

