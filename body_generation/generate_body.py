# MIND_2000: --mind_dataset_subdir MIND_2000
# Adressa_2000: --mind_dataset_subdir Adressa_2000
"""
LLM_E: 실행기 LLM
유저의 취향 파악 후 후보 뉴스의 기대 본문 생성
"""

import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from openai import OpenAI

_BODY_GEN_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NAML_DIR = _BODY_GEN_PROJECT_ROOT / "NAML"
if str(_NAML_DIR) not in sys.path:
    sys.path.insert(0, str(_NAML_DIR))
from naml_dataset_env import DATASET_FILE_PRESETS as MIND_DATASET_PRESETS_BG
# 유저당 여러 후보 시 동시 API 요청 수 (RPM 한도 초과 시 이 값을 코드에서만 줄이면 됨)
DEFAULT_API_CONCURRENCY = 8

# 기본 데이터셋 폴더: dataset/<이 이름>/ 아래 TSV를 읽고, output/<이 이름>/trainN 에 저장
# 우선순위: --mind_dataset_subdir / BodyGenerator 인자 > 환경변수 MIND_DATASET_SUBDIR > 아래 값
DEFAULT_MIND_DATASET_SUBDIR = "MIND_2000"

def _discover_mind_tsv_bg(subdir: str):
    base = _BODY_GEN_PROJECT_ROOT / "dataset" / subdir
    if not base.is_dir():
        return None

    def _pick_news():
        for fixed in ("MIND_news.tsv", "Adressa_news.tsv"):
            p = base / fixed
            if p.is_file():
                return fixed
        cand = sorted(base.glob("*news*.tsv"))
        if len(cand) == 1:
            return cand[0].name
        return None

    def _no_final(paths):
        return [p for p in paths if "_final" not in p.name.lower()]

    news_name = _pick_news()
    if not news_name:
        return None
    trains = sorted(base.glob("MIND_train_*.tsv"))
    tests = sorted(base.glob("MIND_test_*.tsv"))
    if len(tests) > 1:
        tests = _no_final(tests)
    if len(trains) != 1 or len(tests) != 1:
        trains = sorted(base.glob("*_train_*.tsv"))
        tests = sorted(base.glob("*_test_*.tsv"))
        if len(tests) > 1:
            tests = _no_final(tests)
    if len(trains) != 1 or len(tests) != 1:
        return None
    return news_name, trains[0].name, tests[0].name


def _resolve_mind_filenames_bg(subdir: str):
    if subdir in MIND_DATASET_PRESETS_BG:
        n, tr, te = MIND_DATASET_PRESETS_BG[subdir]
    else:
        disc = _discover_mind_tsv_bg(subdir)
        if disc:
            n, tr, te = disc
        else:
            n, tr, te = "MIND_news.tsv", "MIND_train_(1000).tsv", "MIND_test_(1000).tsv"
    if "MIND_NEWS_FILENAME" in os.environ:
        n = os.environ["MIND_NEWS_FILENAME"]
    if "MIND_TRAIN_FILENAME" in os.environ:
        tr = os.environ["MIND_TRAIN_FILENAME"]
    if "MIND_TEST_FILENAME" in os.environ:
        te = os.environ["MIND_TEST_FILENAME"]
    return n, tr, te


def _resolve_mind_dataset_subdir(mind_dataset_subdir: Optional[str] = None) -> str:
    """dataset 하위 폴더명 (예: MIND, MIND_2000).
    우선순위: 인자 > env MIND_DATASET_SUBDIR > DEFAULT_MIND_DATASET_SUBDIR(파일 상단).
    """
    if mind_dataset_subdir:
        return mind_dataset_subdir
    return os.environ.get("MIND_DATASET_SUBDIR") or DEFAULT_MIND_DATASET_SUBDIR


def _default_mind_tsv(filename: str, subdir: Optional[str] = None) -> str:
    s = subdir or _resolve_mind_dataset_subdir()
    return str(_BODY_GEN_PROJECT_ROOT / "dataset" / s / filename)


class BodyGenerator:
    def __init__(self, 
                 prompt_path: str = "body_generation/prompt.yaml",
                 settings_path: str = "body_generation/generation_settings.yaml",
                 news_data_path: Optional[str] = None,
                 train_data_path: Optional[str] = None,
                 test_data_path: Optional[str] = None,
                 use_test: bool = False,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 coordinator_output_dir: str = "coordinator_LLM/output",
                 coordinator_policy_n: Optional[int] = None,
                 coordinator_policy_path: Optional[str] = None,
                 mind_dataset_subdir: Optional[str] = None):
        """
        Args:
            prompt_path: 프롬프트 YAML 파일 경로
            settings_path: 생성 설정 YAML 파일 경로
            news_data_path: 뉴스 데이터 TSV (None이면 dataset/<mind_dataset_subdir>/MIND_news.tsv)
            train_data_path: 학습 TSV (None이면 .../MIND_train_(1000).tsv)
            test_data_path: 테스트 TSV (None이면 .../MIND_test_(1000).tsv)
            mind_dataset_subdir: dataset 하위 폴더명 (예: MIND, MIND_1000). None이면 env 또는 DEFAULT_MIND_DATASET_SUBDIR (저장: self.mind_dataset_subdir)
            use_test: True면 test 데이터 사용, False면 train 데이터 사용
            api_key: OpenAI API 키 (없으면 환경변수 OPENAI_API_KEY 사용)
            model: 사용할 모델명
            coordinator_output_dir: coordinator_LLM 출력 디렉토리 (Tone 등 설정을 N.txt에서 로드)
            coordinator_policy_n: 사용할 정책 파일 번호 (N.txt). None이면 가장 큰 N 사용.
            coordinator_policy_path: JSON 정책 파일 경로 (지정 시 N.txt보다 우선). coordinator 출력과 동일 형식(updated_policy/policy).
        """
        sub = _resolve_mind_dataset_subdir(mind_dataset_subdir)
        self.mind_dataset_subdir = sub
        n_fn, tr_fn, te_fn = _resolve_mind_filenames_bg(sub)
        if news_data_path is None:
            news_data_path = _default_mind_tsv(n_fn, sub)
        if train_data_path is None:
            train_data_path = _default_mind_tsv(tr_fn, sub)
        if test_data_path is None:
            test_data_path = _default_mind_tsv(te_fn, sub)
        self.prompt_path = prompt_path
        self.settings_path = settings_path
        self.news_data_path = news_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.use_test = use_test
        self.model = model
        self.coordinator_output_dir = coordinator_output_dir
        self.coordinator_policy_n = coordinator_policy_n
        self.coordinator_policy_path = coordinator_policy_path
        self._print_lock = threading.Lock()
        
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
        """coordinator JSON(N.txt 또는 임의 경로)에서 정책 로드. coordinator_policy_path가 있으면 최우선."""
        self.coordinator_policy = None
        if self.coordinator_policy_path:
            best_path = os.path.abspath(self.coordinator_policy_path)
            if os.path.isfile(best_path):
                try:
                    with open(best_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.coordinator_policy = (
                        data.get("updated_policy") or data.get("current_policy") or data.get("policy")
                    )
                    if self.coordinator_policy:
                        print(
                            f"Coordinator 설정 로드(파일): {best_path} "
                            f"(tone={self.coordinator_policy.get('tone', '?')}, ...)"
                        )
                except Exception as e:
                    print(f"경고: Coordinator 정책 파일 로드 실패 ({best_path}): {e}")
            else:
                print(f"경고: Coordinator 정책 파일 없음: {best_path}")
            return

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

    def set_coordinator_policy_n(self, n: Optional[int]) -> None:
        """coordinator 출력 N.txt 정책을 바꿔 다시 로드 (배치마다 다른 번호를 쓸 때)."""
        self.coordinator_policy_path = None
        self.coordinator_policy_n = n
        self._load_coordinator_policy()

    def set_coordinator_policy_file(self, path: Optional[str]) -> None:
        """임의 경로의 JSON 정책 파일로 전환 (클러스터별 정책 등). path가 None이면 파일 모드를 끄고 coordinator_policy_n도 None이라 정책이 비게 될 수 있음."""
        self.coordinator_policy_path = path
        self.coordinator_policy_n = None
        self._load_coordinator_policy()
    
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

    def _generate_one_candidate_parallel(
        self,
        position: int,
        total: int,
        user_id: int,
        candidate_news_id: str,
        user_history: list,
        output_dir: Optional[str],
        verbose_save: bool = True,
    ) -> Tuple[int, Optional[Dict]]:
        """후보 1건 생성. (원본 순서 position, 결과) 반환. 실패 시 (position, None)."""
        candidate_title = self.news_dict[candidate_news_id]["title"]
        prompt = self._build_prompt(
            user_history=user_history,
            candidate_title=candidate_title,
        )
        with self._print_lock:
            print(
                f"\n[{position + 1}/{total}] 후보 뉴스 '{candidate_title}' 처리 중... (user={user_id}, id={candidate_news_id})"
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            generated_body = response.choices[0].message.content.strip()
            result = {
                "user_id": user_id,
                "candidate_news_id": candidate_news_id,
                "candidate_title": candidate_title,
                "user_history_count": len(user_history),
                "user_history": user_history,
                "prompt": prompt,
                "generated_body": generated_body,
                "model": self.model,
            }
            if output_dir:
                user_dir = os.path.join(output_dir, f"user_{user_id}")
                os.makedirs(user_dir, exist_ok=True)
                save_path = os.path.join(user_dir, f"news_{candidate_news_id}.json")
                self._save_result(result, save_path, verbose=verbose_save)
            with self._print_lock:
                print(f"[{position + 1}/{total}] 완료: {candidate_news_id}")
            return (position, result)
        except Exception as e:
            with self._print_lock:
                print(f"[{position + 1}/{total}] 오류 ({candidate_news_id}): {e}")
            return (position, None)
    
    def generate_bodies_for_user(self,
                                 user_id: int,
                                 output_dir: Optional[str] = None) -> List[Dict]:
        """
        특정 유저의 모든 candidate_news에 대해 기대 본문 생성
        
        Args:
            user_id: 유저 ID
            output_dir: 결과 저장 디렉토리 (선택, 없으면 저장하지 않음)
        
        Returns:
            생성 결과 리스트 (후보 순서 유지, 성공한 항목만)
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
        
        if not unique_candidate_news_ids:
            print(f"유저 {user_id}: 생성할 후보 뉴스가 없습니다. 건너뜁니다.")
            return []
        
        n_cand = len(unique_candidate_news_ids)
        
        work_positions: List[int] = []
        work_ids: List[str] = []
        for position, candidate_news_id in enumerate(unique_candidate_news_ids):
            if candidate_news_id not in self.news_dict:
                print(f"경고: 뉴스 ID {candidate_news_id}를 찾을 수 없습니다. 건너뜁니다.")
                continue
            work_positions.append(position)
            work_ids.append(candidate_news_id)
        
        if not work_ids:
            print(f"유저 {user_id}: 유효한 후보 뉴스가 없습니다.")
            return []
        
        print(
            f"\n유저 {user_id}에 대해 {len(work_ids)}개의 후보 뉴스에 대한 본문을 생성합니다..."
            f" (병렬, 최대 동시 {min(DEFAULT_API_CONCURRENCY, len(work_ids))}요청)"
        )
        
        max_workers = max(1, min(DEFAULT_API_CONCURRENCY, len(work_ids)))
        verbose_save = max_workers <= 1
        pairs: List[Tuple[int, Optional[Dict]]] = []
        if max_workers == 1:
            for pos, cid in zip(work_positions, work_ids):
                _, res = self._generate_one_candidate_parallel(
                    pos, n_cand, user_id, cid, user_history, output_dir, verbose_save
                )
                pairs.append((pos, res))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(
                        self._generate_one_candidate_parallel,
                        pos,
                        n_cand,
                        user_id,
                        cid,
                        user_history,
                        output_dir,
                        False,
                    )
                    for pos, cid in zip(work_positions, work_ids)
                ]
                for fut in as_completed(futs):
                    pairs.append(fut.result())
        
        pairs.sort(key=lambda x: x[0])
        results = [r for _, r in pairs if r is not None]
        
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

    def generate_bodies_for_pairs(
        self,
        pairs: List[Tuple[int, str]],
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        (user_id, candidate_news_id) 목록에 대해 기대 본문 생성 (병렬).
        train_df에 없는 user는 건너뜀.
        """
        if not pairs:
            print("생성할 (user, candidate) 쌍이 없습니다.")
            return []

        # 유효 후보만, 순서 유지하며 중복 제거
        seen: Set[Tuple[int, str]] = set()
        work: List[Tuple[int, str]] = []
        for uid, cid in pairs:
            cid = str(cid).strip()
            if not cid or cid not in self.news_dict:
                if cid and cid not in self.news_dict:
                    print(f"경고: 뉴스 ID {cid} 없음 — 건너뜀 (user={uid})")
                continue
            key = (int(uid), cid)
            if key in seen:
                continue
            seen.add(key)
            work.append((int(uid), cid))

        if not work:
            print("유효한 후보가 없습니다.")
            return []

        hist_cache: Dict[int, List[str]] = {}

        def _hist(uid: int) -> List[str]:
            if uid not in hist_cache:
                hist_cache[uid] = self._get_user_click_history(uid, max_items=10)
            return hist_cache[uid]

        n_total = len(work)
        print(
            f"\n총 {n_total}개 (user, 후보) 쌍에 대해 본문 생성..."
            f" (병렬, 최대 동시 {min(DEFAULT_API_CONCURRENCY, n_total)}요청)"
        )

        max_workers = max(1, min(DEFAULT_API_CONCURRENCY, n_total))
        verbose_save = max_workers <= 1
        out_pairs: List[Tuple[int, Optional[Dict]]] = []

        def _one(pos: int, uid: int, cid: str) -> Tuple[int, Optional[Dict]]:
            uh = _hist(uid)
            if len(uh) == 0:
                with self._print_lock:
                    print(f"[{pos + 1}/{n_total}] 유저 {uid}: 클릭 히스토리 없음 — 건너뜀")
                return (pos, None)
            return self._generate_one_candidate_parallel(
                pos, n_total, uid, cid, uh, output_dir, verbose_save
            )

        if max_workers == 1:
            for pos, (uid, cid) in enumerate(work):
                out_pairs.append(_one(pos, uid, cid))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(_one, pos, uid, cid)
                    for pos, (uid, cid) in enumerate(work)
                ]
                for fut in as_completed(futs):
                    out_pairs.append(fut.result())

        out_pairs.sort(key=lambda x: x[0])
        results = [r for _, r in out_pairs if r is not None]

        if output_dir and results:
            all_path = os.path.join(output_dir, "all_results_pairs.json")
            with open(all_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n전체 결과 요약: {all_path}")

        print(f"\n총 {len(results)}개의 본문이 생성되었습니다.")
        return results
    
    def _save_result(self, result: Dict, save_path: str, verbose: bool = True):
        """결과를 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if verbose:
            with self._print_lock:
                print(f"결과가 {save_path}에 저장되었습니다.")


def get_next_run_folder(base_output_dir: str, mode: str) -> str:
    """
    생성 정책(대상)에 따라 출력 폴더 경로 반환.
    mode: "train" -> train0, train1, ... (학습 TSV 후보)
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
        raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    """예제 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='뉴스 기대 본문 생성기')
    parser.add_argument('--user_id', type=int, default=None, help='유저 ID (지정하지 않으면 모든 유저 처리)')
    parser.add_argument('--start_user_id', type=int, default=None, help='시작 유저 ID (지정하면 해당 ID부터 이후 모든 유저 처리)')
    parser.add_argument('--candidate_news_id', type=str, default=None, help='후보 뉴스 ID (단일 뉴스 처리용, 없으면 모든 candidate_news 처리)')
    parser.add_argument('--output', type=str, default='body_generation/output',
                        help='출력 루트 (실제 저장은 <루트>/<데이터셋 폴더>/trainN 등, 예: output/MIND_2000/train0)')
    parser.add_argument('--use_test', action='store_true', help='테스트셋 후보로 생성 (미지정 시 학습 TSV 사용)')
    parser.add_argument('--policy_file', type=int, default=None, metavar='N', help='정책으로 사용할 coordinator 출력 파일 번호 (N이면 N.txt). 생략 시 가장 큰 번호 사용')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API 키 (선택, 환경변수 사용 가능)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='사용할 모델명')
    parser.add_argument('--mind_dataset_subdir', type=str, default=None,
                        help='dataset 하위 폴더 (예: MIND, MIND_1000, MIND_2000). 미지정 시 env MIND_DATASET_SUBDIR 또는 MIND')
    args = parser.parse_args()
    
    # 데이터셋별로 출력 분리: body_generation/output/MIND_2000/train0 형태
    dataset_subdir = _resolve_mind_dataset_subdir(args.mind_dataset_subdir)
    base_output_dir = os.path.join(os.path.normpath(args.output), dataset_subdir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 생성 정책에 따라 출력 폴더 결정
    if args.use_test:
        mode = "test"
        run_dir = get_next_run_folder(base_output_dir, mode)
        print(f"데이터셋: {dataset_subdir}")
        print(f"생성 정책: {mode}, 저장 경로: {run_dir}")
    else:
        mode = "train"
        run_dir = get_next_run_folder(base_output_dir, mode)
        print(f"데이터셋: {dataset_subdir}")
        print(f"생성 정책: {mode}, 저장 경로: {run_dir}")
    
    # 생성기 초기화
    generator = BodyGenerator(
        api_key=args.api_key,
        model=args.model,
        use_test=args.use_test,
        coordinator_policy_n=args.policy_file,
        mind_dataset_subdir=args.mind_dataset_subdir,
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

