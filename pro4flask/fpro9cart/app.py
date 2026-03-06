from flask import Flask, render_template, request, session, redirect, url_for
from datetime import timedelta 

app = Flask(__name__)

app.secret_key = "abcdef123456"
app.permanent_session_lifetime = timedelta(minutes=5) # 세션 만료 5분

products = [
    {"id":1, "name":"노트북", "price":3500000},
    {"id":2, "name":"키보드", "price":50000},
    {"id":3, "name":"마우스", "price":35000},
    {"id":4, "name":"모니터", "price":1500000}
]

@app.route('/')
def product_list():
    return render_template("products.html", products=products)

@app.route('/cart')
def show_cart():
    cart = session.get("cart",{})
    total = sum(info["price"] * info["qty"] for info in cart.values())
    return render_template("cart.html", cart=cart, total=total)

@app.route('/add/<int:product_id>')
def add_to_cart(product_id):
    # print(product_id)
    # 세션 cart가 없으면 빈 dict로 생성
    cart = session.get("cart",{})
    # next(..., None)  : 묶음형 자료에서 다음 값 1개를 꺼내는 함수
    # 주문 상품이 product에 기억됨
    product = next((p for p in products if p["id"] == product_id),None)

    if product is None:
        return "상품을 찾을 수 없어요", 404
    
    # 주문 상품이 상품목록에 있으면 장바구니에 추가
    item_name = product["name"]
    
    if item_name in cart:
        cart[item_name]["qty"] += 1   # 카드에 동일 상품이 있는 경우 수량만 증가
    else:
        cart[item_name] = {"price": product["price"], "qty":1} 
        # 카트에 최조 상품일 경우 수량 1(qty 요소(key) 생성)

    session["cart"] = cart   # 변수 cart를 세션에 "cart" 키에 값으로 저장
    session.permanent = True # 5분 만료 적용 

    return redirect(url_for("show_cart"))   # cart에 저장 후 장바구니 보기로 이동

# 장바구니 부분 삭제
@app.route('/remove/<item_name>')
def remove_to_cart(item_name):
    cart = session.get("cart")

    if item_name in cart:
        del cart[item_name]

    session["cart"] = cart
    return redirect(url_for("show_cart"))

# 장바구니 비우기
@app.route('/clear')
def clear_cart():
    session.pop("cart",None)  # 세션에 여러개의 키 중 "cart"라는 키를 삭제
    return redirect(url_for("show_cart"))


if __name__ == '__main__':
    app.run(debug=True)