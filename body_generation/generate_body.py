"""
LLM_E: 실행기 LLM
유저의 취향 파악 후 후보 뉴스의 기대 본문 생성
"""

import os
import math
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
                 model: str = "gpt-4o-mini",
                 coordinator_output_dir: str = "coordinator_LLM/output",
                 train80_only: bool = False,
                 train20_only: bool = False,
                 train20_per_user: bool = False,
                 train20_positive_only: bool = False,
                 train20_first_k: Optional[int] = None,
                 train20_batch_index: int = 0,
                 train80_first_k: Optional[int] = None,
                 train80_batch_index: int = 0,
                 coordinator_policy_n: Optional[int] = None):
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
            coordinator_output_dir: coordinator_LLM 출력 디렉토리 (Tone 등 설정을 N.txt에서 로드)
            train80_only: True면 트레이닝셋 앞 80%에 등장하는 (유저, 후보뉴스)에 대해서만 생성 (use_test=False일 때만 적용)
            train20_only: True면 트레이닝셋 뒤 20%에 등장하는 (유저, 후보뉴스)에 대해서만 생성 (use_test=False일 때만 적용)
            train20_per_user: train20_only일 때 True면 유저별 후반 20% 세션(유저당 최소 1세션)만 사용. False면 전체 행 기준 후반 20%.
            train20_positive_only: train20_only일 때 True면 세션당 positive(클릭된) 후보 1개에만 기대본문 생성. False면 5개 후보 모두.
            train20_first_k: train20_only(유저별 후반 20%)일 때 배치당 세션 수 (예: 500). None이면 전체 사용.
            train20_batch_index: train20_first_k 사용 시 배치 번호 (0=첫 K세션, 1=다음 K세션). 출력 train20_batch{N}.
            train80_first_k: train80_only일 때 배치당 세션 수 (예: 500). None이면 전체 80% 사용.
            train80_batch_index: train80_first_k 사용 시 배치 번호 (0=첫 500세션, 1=다음 500세션). 출력 train80_batch{N}.
            coordinator_policy_n: 사용할 정책 파일 번호 (N.txt). None이면 가장 큰 N 사용.
        """
        self.prompt_path = prompt_path
        self.settings_path = settings_path
        self.news_data_path = news_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.use_test = use_test
        self.model = model
        self.coordinator_output_dir = coordinator_output_dir
        self.train80_only = train80_only
        self.train20_only = train20_only
        self.train20_per_user = train20_per_user
        self.train20_positive_only = train20_positive_only
        self.train20_first_k = train20_first_k
        self.train20_batch_index = train20_batch_index
        self.train80_first_k = train80_first_k
        self.train80_batch_index = train80_batch_index
        self.coordinator_policy_n = coordinator_policy_n
        
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
        
        # coordinator 출력에서 Tone 등 설정 로드 (가장 숫자가 큰 N.txt)
        self._load_coordinator_policy()
    
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
        
        # train80_only: 트레이닝셋 앞 80%에 등장하는 (user_id, candidate_news_id)만 허용 (학습 데이터일 때만)
        self.allowed_train80_pairs = None
        if self.train80_only and not self.use_test and len(self.train_df) > 0:
            n80 = max(1, int(0.8 * len(self.train_df)))
            if self.train80_first_k is not None:
                start = self.train80_batch_index * self.train80_first_k
                end = min(start + self.train80_first_k, n80)
                first80 = self.train_df.iloc[start:end]
                print(f"트레이닝셋 앞 80% 중 배치 {self.train80_batch_index}: 세션 {start}~{end-1} ({end-start}개, train80_first_k={self.train80_first_k})")
            else:
                first80 = self.train_df.iloc[:n80]
            allowed = set()
            for _, row in first80.iterrows():
                uid = row['user']
                if pd.isna(uid):
                    continue
                uid = int(uid)
                for cand_id in str(row['candidate_news']).split():
                    if cand_id.strip():
                        allowed.add((uid, cand_id.strip()))
            self.allowed_train80_pairs = allowed
            print(f"트레이닝셋 앞 80% 후보만 생성: 허용 (user, candidate_news) 쌍 {len(self.allowed_train80_pairs)}개")
        
        # train20_only: 트레이닝셋 뒤 20%에 등장하는 (user_id, candidate_news_id)만 허용 (학습 데이터일 때만)
        self.allowed_train20_pairs = None
        if self.train20_only and not self.use_test and len(self.train_df) > 0:
            def _add_row_pairs(allowed: set, row, uid: int) -> None:
                cand_ids = [c.strip() for c in str(row['candidate_news']).split() if c.strip()]
                clicked_str = str(row.get('clicked', ''))
                clicked_list = clicked_str.split() if clicked_str else []
                if self.train20_positive_only and clicked_list:
                    # 세션당 positive(클릭된) 후보 1개만 추가
                    for i, cid in enumerate(cand_ids):
                        if i < len(clicked_list) and clicked_list[i].strip() == '1':
                            allowed.add((uid, cid))
                            break
                else:
                    for cid in cand_ids:
                        allowed.add((uid, cid))
            if self.train20_per_user:
                # 유저별 후반 20% 세션 (유저당 최소 1세션) → 순서 있는 세션 인덱스 리스트
                train20_row_indices = []
                for uid, grp in self.train_df.groupby('user', dropna=False):
                    if pd.isna(uid):
                        continue
                    n = len(grp)
                    take_count = max(1, int(math.ceil(0.2 * n)))
                    last_rows = grp.tail(take_count)
                    for idx in last_rows.index:
                        train20_row_indices.append(idx)
                train20_row_indices.sort()
                total_t20 = len(train20_row_indices)
                # 배치 슬라이스 (train20_first_k 사용 시)
                if self.train20_first_k is not None:
                    start = self.train20_batch_index * self.train20_first_k
                    end = min(start + self.train20_first_k, total_t20)
                    batch_indices = train20_row_indices[start:end]
                    print(f"트레이닝셋 유저별 뒤 20% 중 배치 {self.train20_batch_index}: 세션 {start}~{end-1} ({len(batch_indices)}개, train20_first_k={self.train20_first_k})")
                else:
                    batch_indices = train20_row_indices
                allowed = set()
                for idx in batch_indices:
                    row = self.train_df.loc[idx]
                    uid = row['user']
                    if pd.isna(uid):
                        continue
                    uid = int(uid)
                    _add_row_pairs(allowed, row, uid)
                self.allowed_train20_pairs = allowed
                pos_only = " (positive만)" if self.train20_positive_only else ""
                batch_info = f", 배치 {self.train20_batch_index}" if self.train20_first_k is not None else ""
                print(f"트레이닝셋 유저별 뒤 20% 후보만 생성{pos_only}{batch_info}: 허용 (user, candidate_news) 쌍 {len(self.allowed_train20_pairs)}개")
            else:
                n80 = max(1, int(0.8 * len(self.train_df)))
                last20_indices = list(self.train_df.iloc[n80:].index)
                total_t20 = len(last20_indices)
                if self.train20_first_k is not None:
                    start = self.train20_batch_index * self.train20_first_k
                    end = min(start + self.train20_first_k, total_t20)
                    batch_indices = last20_indices[start:end]
                    print(f"트레이닝셋 뒤 20%(행 기준) 중 배치 {self.train20_batch_index}: 세션 {start}~{end-1} ({len(batch_indices)}개)")
                else:
                    batch_indices = last20_indices
                allowed = set()
                for idx in batch_indices:
                    row = self.train_df.loc[idx]
                    uid = row['user']
                    if pd.isna(uid):
                        continue
                    uid = int(uid)
                    _add_row_pairs(allowed, row, uid)
                self.allowed_train20_pairs = allowed
                pos_only = " (positive만)" if self.train20_positive_only else ""
                batch_info = f", 배치 {self.train20_batch_index}" if self.train20_first_k is not None else ""
                print(f"트레이닝셋 뒤 20% 후보만 생성{pos_only}{batch_info}: 허용 (user, candidate_news) 쌍 {len(self.allowed_train20_pairs)}개")
        
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
    
    def _load_coordinator_policy(self) -> None:
        """coordinator_LLM/output 폴더에서 N.txt를 찾아 정책 로드. coordinator_policy_n이 있으면 해당 번호, 없으면 가장 큰 N."""
        self.coordinator_policy = None
        if not os.path.isdir(self.coordinator_output_dir):
            return
        best_path = None
        if self.coordinator_policy_n is not None:
            path = os.path.join(self.coordinator_output_dir, f"{self.coordinator_policy_n}.txt")
            if os.path.isfile(path):
                best_path = path
        else:
            max_num = -1
            for f in os.listdir(self.coordinator_output_dir):
                if not f.endswith('.txt'):
                    continue
                base = f[:-4]
                try:
                    n = int(base)
                    if n > max_num:
                        max_num = n
                        best_path = os.path.join(self.coordinator_output_dir, f)
                except ValueError:
                    continue
        if best_path is None:
            return
        try:
            with open(best_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.coordinator_policy = data.get("updated_policy") or data.get("current_policy") or data.get("policy")
            if self.coordinator_policy:
                print(f"Coordinator 설정 로드: {best_path} (tone={self.coordinator_policy.get('tone', '?')}, ...)")
        except Exception as e:
            print(f"경고: Coordinator 설정 로드 실패 ({best_path}): {e}")
    
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
        
        # Tone, Abstraction Level 등: coordinator_LLM/output의 가장 큰 N.txt(updated_policy)에서 로드, 없으면 기본값
        # 프롬프트에는 generation_settings.yaml의 구체적 설명을 채움 (설명이 없으면 값만 사용)
        policy = self.coordinator_policy or {}
        defaults = {"tone": "neutral", "abstraction_level": "mixed", "speculation_count": 1, "length_bucket": "medium", "format": "narrative"}
        category_map = {
            "tone": "Tone",
            "abstraction_level": "Abstraction Level",
            "speculation_count": "Speculation Count",
            "length_bucket": "Length Bucket",
            "format": "Format",
        }
        for key in ["tone", "abstraction_level", "speculation_count", "length_bucket", "format"]:
            val = policy.get(key, defaults[key])
            if isinstance(val, (int, float)):
                val_str = str(int(val))
            else:
                val_str = str(val)
            category = category_map.get(key)
            description = ""
            if category and self.settings_dict.get(category) and val_str in self.settings_dict[category]:
                description = (self.settings_dict[category].get(val_str) or "").strip()
            prompt = prompt.replace("{" + key + "}", description if description else val_str)
        
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
        
        if self.allowed_train80_pairs is not None and (user_id, candidate_news_id) not in self.allowed_train80_pairs:
            raise ValueError(f"(user_id={user_id}, candidate_news_id={candidate_news_id})는 트레이닝셋 앞 80%에 포함되지 않아 생성하지 않습니다.")
        if self.allowed_train20_pairs is not None and (user_id, candidate_news_id) not in self.allowed_train20_pairs:
            raise ValueError(f"(user_id={user_id}, candidate_news_id={candidate_news_id})는 트레이닝셋 뒤 20%에 포함되지 않아 생성하지 않습니다.")
        
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
            
            # 결과 저장 (save_path의 부모가 run 폴더 trainN 이면 그 아래 user_X 로 저장)
            if save_path:
                base_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
                user_dir = os.path.join(base_dir, f"user_{user_id}")
                os.makedirs(user_dir, exist_ok=True)
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
        
        # train80_only / train20_only이면 해당 구간에 등장하는 (user_id, 후보)만 남김
        if self.allowed_train80_pairs is not None:
            unique_candidate_news_ids = [c for c in unique_candidate_news_ids if (user_id, c) in self.allowed_train80_pairs]
            print(f"트레이닝셋 앞 80% 필터 적용: {len(unique_candidate_news_ids)}개의 후보 뉴스에 대해 본문을 생성합니다...")
        if self.allowed_train20_pairs is not None:
            unique_candidate_news_ids = [c for c in unique_candidate_news_ids if (user_id, c) in self.allowed_train20_pairs]
            print(f"트레이닝셋 뒤 20% 필터 적용: {len(unique_candidate_news_ids)}개의 후보 뉴스에 대해 본문을 생성합니다...")
        
        if not unique_candidate_news_ids:
            if self.allowed_train80_pairs is not None:
                print(f"유저 {user_id}: 트레이닝셋 앞 80%에 해당하는 후보 뉴스가 없습니다. 건너뜁니다.")
            elif self.allowed_train20_pairs is not None:
                print(f"유저 {user_id}: 트레이닝셋 뒤 20%에 해당하는 후보 뉴스가 없습니다. 건너뜁니다.")
            else:
                print(f"유저 {user_id}: 생성할 후보 뉴스가 없습니다. 건너뜁니다.")
            return []
        
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
                
                # 개별 파일로 저장 (output_dir = run 폴더 trainN, 그 아래 user_X)
                if output_dir:
                    user_dir = os.path.join(output_dir, f"user_{user_id}")
                    os.makedirs(user_dir, exist_ok=True)
                    save_path = os.path.join(user_dir, f"news_{candidate_news_id}.json")
                    self._save_result(result, save_path)
                
                print("완료!")
                
            except Exception as e:
                print(f"오류 발생: {e}")
                continue
        
        # 전체 결과를 하나의 파일로도 저장
        if output_dir and results:
            user_dir = os.path.join(output_dir, f"user_{user_id}")
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


def get_next_run_folder(base_output_dir: str, mode: str) -> str:
    """
    생성 정책(대상)에 따라 출력 폴더 경로 반환.
    mode: "train" -> train0, train1, ... (트레이닝 앞 80%)
          "train20" -> train20_0, train20_1, ... (트레이닝 뒤 20%)
          "test" -> test_0, test_1, ... (테스트셋)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    if mode == "train":
        max_num = -1
        for name in os.listdir(base_output_dir):
            if not os.path.isdir(os.path.join(base_output_dir, name)):
                continue
            if name.startswith("train") and len(name) > 5:
                try:
                    n = int(name[5:])
                    if n > max_num:
                        max_num = n
                except ValueError:
                    continue
        next_num = max_num + 1
        run_dir = os.path.join(base_output_dir, f"train{next_num}")
    elif mode == "train20":
        prefix = "train20_"
        max_num = -1
        for name in os.listdir(base_output_dir):
            if not os.path.isdir(os.path.join(base_output_dir, name)):
                continue
            if name.startswith(prefix):
                try:
                    n = int(name[len(prefix):])
                    if n > max_num:
                        max_num = n
                except ValueError:
                    continue
        next_num = max_num + 1
        run_dir = os.path.join(base_output_dir, f"train20_{next_num}")
    elif mode == "test":
        prefix = "test_"
        max_num = -1
        for name in os.listdir(base_output_dir):
            if not os.path.isdir(os.path.join(base_output_dir, name)):
                continue
            if name.startswith(prefix):
                try:
                    n = int(name[len(prefix):])
                    if n > max_num:
                        max_num = n
                except ValueError:
                    continue
        next_num = max_num + 1
        run_dir = os.path.join(base_output_dir, f"test_{next_num}")
    else:
        raise ValueError(f"mode must be 'train', 'train20', or 'test', got '{mode}'")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    """예제 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='뉴스 기대 본문 생성기')
    parser.add_argument('--user_id', type=int, default=None, help='유저 ID (지정하지 않으면 모든 유저 처리)')
    parser.add_argument('--start_user_id', type=int, default=None, help='시작 유저 ID (지정하면 해당 ID부터 이후 모든 유저 처리)')
    parser.add_argument('--candidate_news_id', type=str, default=None, help='후보 뉴스 ID (단일 뉴스 처리용, 없으면 모든 candidate_news 처리)')
    parser.add_argument('--output', type=str, default='body_generation/output', help='출력 디렉토리 경로')
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument('--train80_only', action='store_true', help='트레이닝셋 앞 80%% 후보만 생성')
    policy_group.add_argument('--train20_only', action='store_true', help='트레이닝셋 뒤 20%% 후보만 생성')
    policy_group.add_argument('--use_test', action='store_true', help='테스트셋 후보로 생성')
    parser.add_argument('--train20_per_user', action='store_true', help='--train20_only일 때 유저별 후반 20%% 세션(유저당 최소 1)만 사용. NAML 트레이닝 후반 20%%와 동일 정의.')
    parser.add_argument('--train20_positive_only', action='store_true', help='--train20_only일 때 세션당 positive(클릭된) 후보 1개에만 기대본문 생성.')
    parser.add_argument('--train20_first_k', type=int, default=None, metavar='K', help='--train20_only일 때 배치당 K세션 (예: 500). 유저별 후반 20%%와 동일 세션 순서. --train20_batch_index와 함께 사용')
    parser.add_argument('--train20_batch_index', type=int, default=0, metavar='N', help='배치 번호 (0=첫 K세션→train20_batch0, 1=다음 K세션→train20_batch1). --train20_first_k와 함께 사용')
    parser.add_argument('--output_subdir', type=str, default=None, metavar='DIR', help='출력 폴더 이름 (예: train_last20). train20_first_k 미사용 시에만 적용. 지정 시 output/DIR에 저장')
    parser.add_argument('--train80_first_k', type=int, default=None, metavar='K', help='--train80_only일 때 배치당 K세션 (예: 500). --train80_batch_index와 함께 사용')
    parser.add_argument('--train80_batch_index', type=int, default=0, metavar='N', help='배치 번호 (0=첫 500세션→train80_batch0, 1=다음 500세션→train80_batch1). --train80_first_k와 함께 사용')
    parser.add_argument('--policy_file', type=int, default=None, metavar='N', help='정책으로 사용할 coordinator 출력 파일 번호 (N이면 N.txt). 생략 시 가장 큰 번호 사용')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API 키 (선택, 환경변수 사용 가능)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='사용할 모델명')
    
    args = parser.parse_args()
    
    # 생성 정책에 따라 출력 폴더 결정
    if args.train80_only:
        mode = "train"
        if args.train80_first_k is not None:
            run_dir = os.path.join(args.output, f"train80_batch{args.train80_batch_index}")
            os.makedirs(run_dir, exist_ok=True)
            print(f"생성 정책: train80 배치 {args.train80_batch_index} ({args.train80_first_k}세션), 저장 경로: {run_dir}")
        else:
            run_dir = get_next_run_folder(args.output, mode)
            print(f"생성 정책: {mode}, 저장 경로: {run_dir}")
    elif args.train20_only:
        mode = "train20"
        if args.train20_first_k is not None:
            run_dir = os.path.join(args.output, f"train20_batch{args.train20_batch_index}")
            os.makedirs(run_dir, exist_ok=True)
            print(f"생성 정책: {mode} 배치 {args.train20_batch_index} ({args.train20_first_k}세션), 저장 경로: {run_dir}")
        elif args.output_subdir:
            run_dir = os.path.join(args.output, args.output_subdir)
            os.makedirs(run_dir, exist_ok=True)
            print(f"생성 정책: {mode} (output_subdir={args.output_subdir}), 저장 경로: {run_dir}")
        else:
            run_dir = get_next_run_folder(args.output, mode)
            print(f"생성 정책: {mode}, 저장 경로: {run_dir}")
    else:
        mode = "test"
        run_dir = get_next_run_folder(args.output, mode)
        print(f"생성 정책: {mode}, 저장 경로: {run_dir}")
    
    # 생성기 초기화
    generator = BodyGenerator(
        api_key=args.api_key,
        model=args.model,
        use_test=args.use_test,
        train80_only=args.train80_only,
        train20_only=args.train20_only,
        train20_per_user=args.train20_per_user,
        train20_positive_only=args.train20_positive_only,
        train20_first_k=args.train20_first_k,
        train20_batch_index=args.train20_batch_index,
        train80_first_k=args.train80_first_k,
        train80_batch_index=args.train80_batch_index,
        coordinator_policy_n=args.policy_file
    )
    
    if args.candidate_news_id:
        # 단일 뉴스 처리 (user_id 필수)
        if args.user_id is None:
            raise ValueError("단일 뉴스 처리를 위해서는 --user_id를 지정해야 합니다.")
        save_path = os.path.join(run_dir, f"news_{args.candidate_news_id}.json")
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
            output_dir=run_dir
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
                    output_dir=run_dir
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

