# pip install flask
from flask import Flask  # 웹서버(Application Server) 생성에 필요.
# Application Server :  python 프로그램 코드를 실행해서 요청을 처리하는 서버

# Flask 기본 서버는 실무용이 아님. 개발용, 학습용 -  Light-weight server
# 실무용 서버(WSGI): gunicorn, waitress, nginx ...

# waitress 서버를 사용한다면 pip install waitress
from waitress import serve

app = Flask(__name__)  # Flask 객체 생성__name__ : 현재 모듈의 이름

@app.route('/')   # URL 매핑(라우팅).  클라이언트 요청이 '/'일때 아래 함수 수행
def abc():  # 클라이언트 요청을 처리하는 함수
    return '<h1>안녕하세요</h1>'

@app.route('/about')  
def about():  
    return '플라스크를 소개하자면'

@app.route('/user/<name>')   # URL에 변수를 담아 요청
def user(name):  
    return f'내 친구 {name}'

if __name__ == '__main__':
    # 학습용 기본 서버
    # app.run()
    # app.run(debug=True, host='127.0.0.1', port=5000)

    # waitress 실습용 서버 사용시
    print('웹서버 서비스 시작...')
    serve(app=app, host='0.0.0.0', port=8000)


