파일 설명
Q.py : 학습에 사용된 에이전트 코드 (Q-learning & UCB exploration)
utils.py : reward 계산 함수 및 에이전트에게 전달할 state를 생성하는 코드
agent_Q.py : 과제 공지에서 제공한 스켈레톤 코드
save.pkl : q테이블을 포함한 기타 테이블이 저장된 파일
requirements.txt : 과제 공지에 명시된 방법에 의거한 requirements.txt 파일

환경 설정
$ pip install -U knu-rl-env
requirements.txt 파일은 과제 규칙에 의거하여 첨부된 파일입니다. 
실제 환경 설정을 위해서는 과제에서 명시된 python 환경에서 위의 명령어를 입력해 주세요.

실행 방법
$ python agent_Q.py
agent_Q.py 내의 메인 함수를 적절히 수정하여 train, load, evaluate를 진행할 수 있습니다.

실행 결과
동일한 환경에서 실행 시 290 iteration에서 최적 경로를 탐색하는 것을 보장하며, 310 iteration에서 UCB value가 적절하게 수렴합니다.