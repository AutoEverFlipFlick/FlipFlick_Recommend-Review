# 베이스 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 종속성(필요하다면) 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 컨테이너 내 포트
EXPOSE 80

# 환경변수: .env는 GitHub Actions 또는 ECS에서 주입
# CMD로 앱 실행
CMD ["python", "app.py"]

