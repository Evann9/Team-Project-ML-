from flask import Flask, render_template, request, redirect, url_for, flash, session
# pip install python-dotenv
from dotenv import load_dotenv
import pymysql
import os
from flask import get_flashed_messages

load_dotenv()   # .env 파일에 저장된 환경변수 읽기 함수

app = Flask(__name__)
app.secret_key = "abcdef123456" # session/flash 쿠키 서명용 비밀키

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",   # 전세계 문자(한글 포함) + 이모지 => 처리 가능
        cursorclass=pymysql.cursors.DictCursor,    
        autocommit=True
    )

@app.get('/')
def root():
    return redirect(url_for("login_form"))

# 로그인 폼
@app.get('/login')
def login_form():
    return render_template("login.html")

@app.post('/login')
def login_post():
    jikwonno_raw = (request.form.get("jikwonno") or "").strip()
    jikwonname = (request.form.get("jikwonname") or "").strip()

    if not jikwonno_raw.isdigit() or not jikwonname:
        flash("직원번호는 숫자, 직원이름은 필수")
        return redirect(url_for("login_form"))
    
    jikwonno = int(jikwonno_raw)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 로그인 체크
            cur.execute("""
                select jikwonno, jikwonname from jikwon
                where jikwonno=%s and jikwonname=%s
            """,(jikwonno,jikwonname))
            me= cur.fetchone()

            if not me:
                flash("로그인 실패:직원정보 불일치")
                return redirect(url_for("login_form"))
            
            # 로그인 성공인 경우
            cur.execute("""
                select jikwonno, jikwonname, busername, jikwonjik, jikwonpay, year(jikwonibsail) as jikwonibsail_year 
                from jikwon inner join buser on jikwon.busernum = buser.buserno
                order by jikwonno
            """)
            rows= cur.fetchall()

        # 세션 생성
        session["jikwonno"] = me["jikwonno"]
        session["jikwonname"] = me["jikwonname"]
        
        return render_template("jikwonlist.html", rows=rows, login_user=me)

    finally:
        conn.close()

@app.get("/gogek/<int:jikwonno>")
def gogek_list(jikwonno:int):
    if "jikwonno" not in session:
        flash("로그인 후 고객 정보 이용하세요")
        return redirect(url_for("login_form"))
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select gogekno,gogekname,gogektel
                from gogek
                where gogekdamsano = %s order by gogekno
            """,(jikwonno,))
            customers = cur.fetchall()

            cur.execute("""
                select jikwonname from jikwon
                where jikwonno = %s
            """,(jikwonno,))
            emp = cur.fetchone()

        return render_template("gogeklist.html", customers=customers, empno=jikwonno, empname=(emp["jikwonname"] if emp else ""))

    finally:
        conn.close()

@app.get("/jikwons")
def jikwon_list():
    if "jikwonno" not in session:
        flash("로그인 후 이용하세요")
        return redirect(url_for("login_form"))
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select jikwonno, jikwonname, busername, jikwonjik, jikwonpay, 
                year(jikwonibsail) as jikwonibsail_year 
                from jikwon inner join buser on jikwon.busernum = buser.buserno
                order by jikwonno
            """)
            rows= cur.fetchall()
        login_user = {"jikwonno":session['jikwonno'],"jikwonname":session["jikwonname"]}
        return render_template("jikwonlist.html",rows=rows, login_user=login_user)
    
    finally:
        conn.close()

# 로그아웃
@app.post('/logout')
def logout():
    session.clear()
    return redirect(url_for("login_form"))

if __name__ == '__main__':
    app.run(debug=True)