# src/main.py
import argparse
import sys, os

# 프로젝트 루트(/app)가 PYTHONPATH에 없을 수도 있어 대비
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["naver", "petronet", "renewl"], required=True)

    # 공통
    parser.add_argument("--full", action="store_true", help="끝 페이지까지 전부 수집")
    parser.add_argument("--pages", type=int, default=3, help="최신 N페이지만 수집")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--use-hash", action="store_true")

    # 환경설정 전달(각 모듈이 parse_known_args로 읽어감)
    parser.add_argument("--profile", help="환경 프로필(.env_{profile} 또는 .env.{profile})")
    parser.add_argument("--env-file", help="임의 경로의 env 파일(.env1 등)")

    # 네이버 전용
    parser.add_argument("--upjong", type=str, default="%BF%A1%B3%CA%C1%F6", help="네이버 업종 코드")

    # (선택) petronet 확장 옵션
    parser.add_argument("--reset-state", action="store_true", help="petronet 상태파일 삭제 후 시작")
    parser.add_argument("--force", action="store_true", help="petronet 상태 무시하고 전부 시도")
    parser.add_argument("--no-stop-on-repeat", action="store_true", help="첫 링크 반복 감지해도 중단하지 않음")

    args = parser.parse_args()

    # 여기서 모듈을 import (lazy import) 해야 각 모듈의 parse_known_args가 위 인자를 볼 수 있음
    if args.site == "naver":
        from src.naver_research import naver_main  # 또는 src.naver_research 에 실제 함수가 있으면 거기로
        naver_main(
            full=args.full,
            pages=args.pages,
            start=args.start,
            end=args.end,
            use_hash=args.use_hash,
            upjong=args.upjong,
        )
    elif args.site == "renewl":
        from src.renewl_research import renewl_main
        renewl_main(
            pages=args.pages,
        )
    else:  # petronet
        from src.petronet_crawling import petronet_main
        petronet_main(
            full=args.full,
            pages=args.pages,
            start=args.start,
            end=args.end,
            use_hash=args.use_hash,
            # 아래 3개 옵션을 petronet_crawling에서 사용하도록 구현했다면 인자로 넘겨도 됨
            # reset_state=args.reset_state,
            # force=args.force,
            # no_stop_on_repeat=args.no_stop_on_repeat,
        )

if __name__ == "__main__":
    main()
