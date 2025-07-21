from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import requests
import schedule
import time
import threading
from datetime import datetime, timedelta
import redis
import json
import logging
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = Flask(__name__)

# 로깅 설정
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=log_level)

# 환경변수에서 설정 로드
class Config:
    # Spring Boot 서버 설정
    SPRING_SERVER_URL = os.getenv('SPRING_SERVER_URL')

    # Redis 설정
    REDIS_HOST = os.getenv('REDIS_HOST')
    REDIS_PORT = int(os.getenv('REDIS_PORT'))
    REDIS_DB = int(os.getenv('REDIS_DB'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

    # 캐시 설정
    SIMILARITY_CACHE_KEY = os.getenv('SIMILARITY_CACHE_KEY')
    CACHE_EXPIRE_SECONDS = int(os.getenv('CACHE_EXPIRE_SECONDS'))
    EMPTY_CACHE_EXPIRE_SECONDS = int(os.getenv('EMPTY_CACHE_EXPIRE_SECONDS'))

    # 스케줄러 설정
    SCHEDULE_TIME = os.getenv('SCHEDULE_TIME')

    # 플라스크 실행 포트
    FLASK_PORT = int(os.getenv('FLASK_PORT', 80))

    # 유사도 계산 설정
    SIMPLE_SIMILARITY_THRESHOLD = float(os.getenv('SIMPLE_SIMILARITY_THRESHOLD'))
    ADVANCED_SIMILARITY_THRESHOLD = float(os.getenv('ADVANCED_SIMILARITY_THRESHOLD'))
    RATING_WEIGHT = float(os.getenv('RATING_WEIGHT'))
    GENRE_WEIGHT = float(os.getenv('GENRE_WEIGHT'))
    ADVANCED_RATING_WEIGHT = float(os.getenv('ADVANCED_RATING_WEIGHT'))
    ADVANCED_GENRE_WEIGHT = float(os.getenv('ADVANCED_GENRE_WEIGHT'))

    # API 설정
    API_TIMEOUT = int(os.getenv('API_TIMEOUT'))
    MAX_SIMILAR_USERS = int(os.getenv('MAX_SIMILAR_USERS'))

    # 데이터 임계값 설정
    MIN_USERS_FOR_ADVANCED = int(os.getenv('MIN_USERS_FOR_ADVANCED'))
    MIN_MOVIES_FOR_ADVANCED = int(os.getenv('MIN_MOVIES_FOR_ADVANCED'))


# Redis 클라이언트 초기화
try:
    REDIS_CLIENT = redis.Redis(
        host=Config.REDIS_HOST, 
        port=Config.REDIS_PORT, 
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD,
        decode_responses=True  # 자동으로 문자열 디코딩
    )
    # Redis 연결 테스트
    REDIS_CLIENT.ping()
    logging.info(f"Redis 연결 성공: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
except Exception as e:
    logging.error(f"Redis 연결 실패: {e}")
    REDIS_CLIENT = None

class RecommendationEngine:
    def __init__(self):
        self.user_similarities = {}
        self.last_calculated = None
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        logging.info("추천 시스템 초기화 시작")
        
        if not REDIS_CLIENT:
            logging.warning("Redis 클라이언트가 없어 캐시 초기화를 건너뜁니다.")
            return
        
        # Redis 캐시 초기화
        try:
            # 추천 시스템 관련 키만 삭제
            pattern_keys = [
                Config.SIMILARITY_CACHE_KEY,
                'user_profile_*', 
                'recommendation_*'
            ]
            
            for pattern in pattern_keys:
                keys = REDIS_CLIENT.keys(pattern)
                if keys:
                    REDIS_CLIENT.delete(*keys)
                    logging.info(f"Redis 패턴 '{pattern}' 삭제: {len(keys)}개")
            
        except Exception as e:
            logging.warning(f"Redis 초기화 실패: {e}")
        
        # 시스템 상태 초기화
        self.user_similarities = {}
        self.last_calculated = None
        
        logging.info("추천 시스템 초기화 완료")

    def fetch_data_from_spring(self):
        """Spring 서버에서 데이터 가져오기"""
        try:
            url = f"{Config.SPRING_SERVER_URL}/api/v1/recommendation/export-data"
            logging.info(f"Spring 서버에 요청: {url}")
            
            response = requests.get(url, timeout=Config.API_TIMEOUT)
            logging.info(f"응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logging.info(f"받은 데이터 수: {len(data)}건")
                
                if len(data) > 0:
                    logging.info(f"데이터 샘플: {data[0]}")
                    return pd.DataFrame(data)
                else:
                    logging.warning("Spring에서 빈 데이터 반환")
                    return pd.DataFrame()
            else:
                logging.error(f"Spring 서버 오류: {response.status_code}")
                return pd.DataFrame()
                
        except requests.exceptions.ConnectionError:
            logging.error("Spring 서버 연결 실패")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"데이터 가져오기 실패: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """데이터 전처리 - 장르 배열 처리"""
        # 1. 사용자-영화 매트릭스 생성 (평점 기반)
        user_movie_matrix = df.pivot_table(
            index='memberId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # 2. 장르 배열을 개별 행으로 확장
        genre_expanded = []
        for _, row in df.iterrows():
            if isinstance(row['genres'], list):
                for genre in row['genres']:
                    genre_expanded.append({
                        'memberId': row['memberId'],
                        'genre': genre,
                        'rating': row['rating']
                    })
            else:
                # 혹시 문자열로 온 경우 처리
                genre_expanded.append({
                    'memberId': row['memberId'],
                    'genre': row['genres'],
                    'rating': row['rating']
                })
        
        genre_df = pd.DataFrame(genre_expanded)
        
        # 3. 장르 기반 사용자 프로필 생성 (사용자별 장르별 평균 평점)
        if not genre_df.empty:
            genre_profile = genre_df.groupby(['memberId', 'genre'])['rating'].mean().unstack(fill_value=0)
        else:
            genre_profile = pd.DataFrame()
        
        logging.info(f"사용자-영화 매트릭스 크기: {user_movie_matrix.shape}")
        logging.info(f"장르 프로필 크기: {genre_profile.shape}")
        
        return user_movie_matrix, genre_profile
    
    def calculate_similarities(self, user_movie_matrix, genre_profile):
        """개선된 유사도 계산 - 데이터 양에 따른 적응형"""
        similarities = {}
        user_ids = user_movie_matrix.index.tolist()
        total_users = len(user_ids)
        
        logging.info(f"유사도 계산 시작: {total_users}명의 사용자")
        logging.info(f"사용자-영화 매트릭스:\n{user_movie_matrix}")
        
        # 데이터 양에 따른 방법 선택
        if total_users < Config.MIN_USERS_FOR_ADVANCED or user_movie_matrix.shape[1] < Config.MIN_MOVIES_FOR_ADVANCED:
            return self.calculate_simple_similarities(user_movie_matrix, genre_profile)
        else:
            return self.calculate_advanced_similarities(user_movie_matrix, genre_profile)

    def calculate_simple_similarities(self, user_movie_matrix, genre_profile):
        """적은 데이터용 단순 유사도 계산"""
        similarities = {}
        user_ids = user_movie_matrix.index.tolist()
        
        logging.info("단순 유사도 계산 모드 사용")
        
        for i, user_id in enumerate(user_ids):
            user_similar_list = []
            
            for j, other_user_id in enumerate(user_ids):
                if i != j:
                    # 공통 영화 찾기
                    user_i_ratings = user_movie_matrix.iloc[i]
                    user_j_ratings = user_movie_matrix.iloc[j]
                    
                    common_mask = (user_i_ratings > 0) & (user_j_ratings > 0)
                    common_movies = user_i_ratings[common_mask]
                    other_common = user_j_ratings[common_mask]
                    
                    similarity_score = 0
                    
                    if len(common_movies) > 0:
                        # 평점 차이 기반 유사도
                        rating_diff = np.mean(np.abs(common_movies - other_common))
                        rating_similarity = max(0, 1 - rating_diff / 4)
                        
                        # 장르 유사도 추가
                        genre_similarity = 0
                        if not genre_profile.empty:
                            try:
                                genre_i = genre_profile.iloc[i]
                                genre_j = genre_profile.iloc[j]
                                
                                dot_product = np.dot(genre_i, genre_j)
                                norm_i = np.linalg.norm(genre_i)
                                norm_j = np.linalg.norm(genre_j)
                                
                                if norm_i > 0 and norm_j > 0:
                                    genre_similarity = dot_product / (norm_i * norm_j)
                                
                            except Exception as e:
                                logging.warning(f"장르 유사도 계산 실패: {e}")
                        
                        # 최종 유사도 (환경변수에서 가중치 사용)
                        similarity_score = Config.RATING_WEIGHT * rating_similarity + Config.GENRE_WEIGHT * max(0, genre_similarity)
                    
                    logging.info(f"사용자 {user_id} vs {other_user_id}: "
                               f"공통영화={len(common_movies)}, 유사도={similarity_score:.3f}")
                    
                    # 환경변수에서 임계값 사용
                    if similarity_score > Config.SIMPLE_SIMILARITY_THRESHOLD:
                        user_similar_list.append({
                            'similarUserId': int(other_user_id),
                            'similarityScore': float(similarity_score)
                        })
            
            # 유사도 순 정렬
            user_similar_list.sort(key=lambda x: x['similarityScore'], reverse=True)
            similarities[user_id] = user_similar_list[:Config.MAX_SIMILAR_USERS]
            
            logging.info(f"사용자 {user_id}의 유사 사용자: {len(user_similar_list)}명")
        
        return similarities

    def calculate_advanced_similarities(self, user_movie_matrix, genre_profile):
        """충분한 데이터용 고급 유사도 계산"""
        similarities = {}
        
        logging.info("고급 유사도 계산 모드 사용")
        
        # 평점 기반 유사도 - 조정된 코사인 유사도 사용
        rating_similarity = None
        if not user_movie_matrix.empty and user_movie_matrix.shape[0] > 1:
            try:
                user_means = user_movie_matrix.replace(0, np.nan).mean(axis=1)
                centered_matrix = user_movie_matrix.sub(user_means, axis=0).fillna(0)
                
                rating_similarity = cosine_similarity(centered_matrix)
                logging.info("조정된 코사인 유사도 계산 완료")
            except Exception as e:
                logging.warning(f"평점 유사도 계산 실패: {e}")
        
        # 장르 선호도 기반 유사도
        genre_similarity = None
        if not genre_profile.empty and genre_profile.shape[0] > 1:
            try:
                genre_similarity = cosine_similarity(genre_profile.fillna(0))
                logging.info("장르 기반 유사도 계산 완료")
            except Exception as e:
                logging.warning(f"장르 유사도 계산 실패: {e}")
        
        # 최종 유사도 계산 (환경변수에서 가중치 사용)
        final_similarity = None
        if rating_similarity is not None and genre_similarity is not None:
            final_similarity = Config.ADVANCED_RATING_WEIGHT * rating_similarity + Config.ADVANCED_GENRE_WEIGHT * genre_similarity
            logging.info("평점+장르 결합 유사도 사용")
        else:
            logging.warning("유사도 계산 불가능")
            return similarities
        
        # 사용자 ID와 매핑
        user_ids = user_movie_matrix.index.tolist()
        
        for i, user_id in enumerate(user_ids):
            user_similarities = final_similarity[i]
            similar_indices = np.argsort(user_similarities)[::-1]
            
            user_similar_list = []
            for idx in similar_indices:
                if idx != i and user_similarities[idx] > Config.ADVANCED_SIMILARITY_THRESHOLD:
                    user_similar_list.append({
                        'similarUserId': int(user_ids[idx]),
                        'similarityScore': float(user_similarities[idx])
                    })
                    
                    if len(user_similar_list) >= Config.MAX_SIMILAR_USERS:
                        break
            
            similarities[user_id] = user_similar_list
        
        return similarities
    
    def save_similarities_to_spring(self, similarities):
        """계산된 유사도를 Spring 서버에 저장"""
        try:
            # 기존 데이터 삭제 요청
            delete_response = requests.delete(
                f"{Config.SPRING_SERVER_URL}/api/v1/recommendation/similarities",
                timeout=Config.API_TIMEOUT
            )
            logging.info(f"기존 데이터 삭제 응답: {delete_response.status_code}")
            
            # 새 데이터 저장
            for user_id, similar_users in similarities.items():
                payload = {
                    'memberId': int(user_id),
                    'similarities': similar_users
                }
                
                logging.info(f"사용자 {user_id} 데이터 전송: {payload}")
                
                response = requests.post(
                    f"{Config.SPRING_SERVER_URL}/api/v1/recommendation/similarities",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=Config.API_TIMEOUT
                )
                
                if response.status_code == 200:
                    logging.info(f"사용자 {user_id} 유사도 저장 성공")
                else:
                    logging.error(f"유사도 저장 실패 (사용자 {user_id}): {response.status_code}")
                    logging.error(f"응답 내용: {response.text}")
            
            logging.info("유사도 데이터 저장 완료")
            
        except Exception as e:
            logging.error(f"유사도 저장 실패: {e}")
    
    def calculate_and_cache_similarities(self):
        """유사도 계산 및 캐싱"""
        logging.info("유사도 계산 시작...")
        
        # 데이터 가져오기
        df = self.fetch_data_from_spring()
        if df is None or df.empty:
            logging.warning("사용할 데이터가 없습니다. 유사도 계산을 건너뜁니다.")
            
            # 빈 유사도 데이터로 캐시 설정
            empty_similarities = {}
            if REDIS_CLIENT:
                REDIS_CLIENT.setex(
                    Config.SIMILARITY_CACHE_KEY, 
                    Config.EMPTY_CACHE_EXPIRE_SECONDS,
                    json.dumps(empty_similarities)
                )
            
            self.user_similarities = empty_similarities
            self.last_calculated = datetime.now()
            
            logging.info("빈 유사도 데이터로 캐시 설정 완료")
            return
        
        # 데이터 전처리
        user_movie_matrix, genre_profile = self.preprocess_data(df)
        
        # 유사도 계산
        similarities = self.calculate_similarities(user_movie_matrix, genre_profile)
        
        # Redis에 캐싱
        if REDIS_CLIENT:
            REDIS_CLIENT.setex(
                Config.SIMILARITY_CACHE_KEY, 
                Config.CACHE_EXPIRE_SECONDS,
                json.dumps(similarities)
            )
            logging.info("새로운 유사도 데이터 캐시 저장 완료")
        
        # Spring 서버에 저장
        self.save_similarities_to_spring(similarities)
        
        self.user_similarities = similarities
        self.last_calculated = datetime.now()
        
        logging.info(f"유사도 계산 완료: {len(similarities)}명의 사용자")

# 추천 엔진 인스턴스
recommendation_engine = RecommendationEngine()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'last_calculated': recommendation_engine.last_calculated.isoformat() if recommendation_engine.last_calculated else None,
        'redis_connected': REDIS_CLIENT is not None,
        'config': {
            'spring_url': Config.SPRING_SERVER_URL,
            'redis_host': Config.REDIS_HOST,
            'schedule_time': Config.SCHEDULE_TIME
        }
    }), 200

@app.route('/recalculate-similarity', methods=['POST'])
def recalculate_similarity():
    """유사도 재계산 API"""
    try:
        # 백그라운드에서 실행
        thread = threading.Thread(target=recommendation_engine.calculate_and_cache_similarities)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '유사도 재계산이 시작되었습니다.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/similar-users/<int:user_id>', methods=['GET'])
def get_similar_users(user_id):
    """특정 사용자의 유사 사용자 조회"""
    try:
        if not REDIS_CLIENT:
            return jsonify({
                'user_id': user_id,
                'similar_users': [],
                'cached': False,
                'message': 'Redis 연결이 없습니다.'
            })
        
        # Redis에서 캐시된 데이터 조회
        cached_data = REDIS_CLIENT.get(Config.SIMILARITY_CACHE_KEY)
        if cached_data:
            similarities = json.loads(cached_data)
            similar_users = similarities.get(str(user_id), [])
            
            return jsonify({
                'user_id': user_id,
                'similar_users': similar_users,
                'cached': True
            })
        else:
            return jsonify({
                'user_id': user_id,
                'similar_users': [],
                'cached': False,
                'message': '캐시된 데이터가 없습니다. 재계산이 필요합니다.'
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/user-profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """사용자의 장르 선호도 프로필 조회"""
    try:
        # 최신 데이터로 프로필 계산
        df = recommendation_engine.fetch_data_from_spring()
        if df is None or df.empty:
            return jsonify({'error': '데이터가 없습니다.'}), 404
            
        # 해당 사용자 데이터만 필터링
        user_data = df[df['memberId'] == user_id]
        if user_data.empty:
            return jsonify({'error': '사용자 데이터가 없습니다.'}), 404
            
        # 장르별 평균 평점 계산
        genre_ratings = {}
        total_ratings = []
        
        for _, row in user_data.iterrows():
            rating = row['rating']
            total_ratings.append(rating)
            
            if isinstance(row['genres'], list):
                for genre in row['genres']:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(rating)
        
        # 장르별 평균 계산
        genre_avg = {genre: np.mean(ratings) for genre, ratings in genre_ratings.items()}
        
        return jsonify({
            'user_id': user_id,
            'total_movies': len(user_data),
            'average_rating': np.mean(total_ratings),
            'genre_preferences': dict(sorted(genre_avg.items(), key=lambda x: x[1], reverse=True))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/sample-data', methods=['GET'])
def get_sample_data():
    """디버깅용 샘플 데이터 조회"""
    try:
        df = recommendation_engine.fetch_data_from_spring()
        if df is None or df.empty:
            return jsonify({'message': '데이터가 없습니다.'})
            
        sample_data = df.head(5).to_dict('records')
        return jsonify({
            'total_records': len(df),
            'sample_data': sample_data,
            'columns': df.columns.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """현재 설정 조회 (디버깅용)"""
    return jsonify({
        'spring_server_url': Config.SPRING_SERVER_URL,
        'redis_config': {
            'host': Config.REDIS_HOST,
            'port': Config.REDIS_PORT,
            'db': Config.REDIS_DB
        },
        'cache_config': {
            'cache_key': Config.SIMILARITY_CACHE_KEY,
            'expire_seconds': Config.CACHE_EXPIRE_SECONDS
        },
        'algorithm_config': {
            'simple_threshold': Config.SIMPLE_SIMILARITY_THRESHOLD,
            'advanced_threshold': Config.ADVANCED_SIMILARITY_THRESHOLD,
            'rating_weight': Config.RATING_WEIGHT,
            'genre_weight': Config.GENRE_WEIGHT
        },
        'schedule_time': Config.SCHEDULE_TIME
    })

def scheduled_recalculation():
    """스케줄된 재계산"""
    recommendation_engine.calculate_and_cache_similarities()

# 스케줄링 설정 (환경변수에서 시간 로드)
schedule.every().day.at(Config.SCHEDULE_TIME).do(scheduled_recalculation)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    # 서버 시작시 초기 계산
    initial_thread = threading.Thread(target=recommendation_engine.calculate_and_cache_similarities)
    initial_thread.start()
    
    # 스케줄러 시작
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Flask 서버 시작
    app.run(host='0.0.0.0', port=Config.FLASK_PORT, debug=False)
