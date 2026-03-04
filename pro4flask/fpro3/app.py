# Flask app의 Entry point
# 라우팅, 서버 실행 담당

# jinja2 : Flask에서 HTML을 동적으로 랜더링할 때 사용하는 템플릿 엔진
# 웹 서버에서 html문을 완성한 후 클라이언트 전송
# html 안에 파이썬 변수를 넣고, 반복/조건문 등을 사용할 수 있게 해주는 도구

# render_template : html 템플릿 파일(jinja2 템플릿)을 읽어 필요한 값을 채운 후
#                   완성된 HTML을 응답으로 반환해주는 함수
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/hello')
def hello():
    name = '길동아 '
    addr = '강남구 테헤란로'
    # 템플릿 변수에 전달할 땐 변수명 = 값 형태로 적어 준다.
    return render_template("hello.html", name=name, juso=addr)

@app.route('/world')
def world_img():
    return render_template("my.html")

if __name__ == '__main__':
    app.run(debug=True)







