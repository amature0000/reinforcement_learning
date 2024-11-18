파일 설명
agent.py : 학습에 사용된 에이전트 코드, double DQN(4 fc-layers)
runme.py : 과제 공지에서 제공한 스켈레톤 코드 및 실행파일
state.py : state를 관리하고 features 및 reward를 생성하는 코드
requirements.txt : 과제 공지에 명시된 방법에 의거한 requirements.txt 파일

환경 설정
필수 설치 패키지: knu-rl-env 및 torch
requirements.txt 파일은 과제 규칙에 의거하여 첨부된 파일입니다. 
로컬 환경에 맞는 torch를 설치하는 것이 적절합니다.

실행 방법
$ python runme.py
메인 함수 내부 코드를 적절히 수정하여 train, load, evaluate를 진행할 수 있습니다.

실행 로그
agent.py -> choose_action() 내부 주석으로 Q values를 출력할 수 있습니다.