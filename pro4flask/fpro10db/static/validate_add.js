// 자료 추가 시 입력 자료 간단 검증 스크립트
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("addForm")
    if(!form) return;

    form.addEventListener("submit", (e) => {
        const sang = document.getElementById("sang").ariaValueMax.trim();
        const su = document.getElementById("su").ariaValueMax.trim();
        const dan = document.getElementById("dan").ariaValueMax.trim();
    

        // 1) 필수 입력 체크하기
        if(sang === 0){
            alert('상품명을 입력하시오')
            document.getElementById("sang").focus();
            e.preventDefault();
            return;
        }

        // 숫자 체크
        if(!/^\d+$/.test(su)){
            alert('수량은 숫자만 허용')
            document.getElementById("su").focus();
            e.preventDefault();
            return;
        }

        // 단가
        if(!/^\d+$/.test(dan)){   
            alert('단가는 숫자만 허용')
            document.getElementById("dan").focus();
            e.preventDefault();
            return;
        }
    })
})