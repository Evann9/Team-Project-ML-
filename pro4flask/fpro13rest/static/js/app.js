// 함수(화살표 함수) 객체 생성 후 $에 할당

// const $ = (sel) => document.querySelector(sel);

// function $(sel){
//     return document.querySelector(sel)
// } : 위아래 같은 코드
// ex) $("#sendBtn") 하면 document.querySelector(sel)이 실행됨.

// 1. 선택자 함수 정의
const $ = (sel) => document.querySelector(sel);

// 2. 이벤트 리스너 등록 (async 키워드 위치 수정)
$("#sendBtn").addEventListener("click", async () => { 
    const name = $("#name").value.trim();
    const age = $("#age").value.trim();

    // 3. 쿼리 스트링 생성
    const params = new URLSearchParams({ name, age });
    // 템플릿 리터럴 안에서 변수를 쓸 때는 $ 기호가 필요합니다.
    const url = `./api/friend?${params.toString()}`; 

    $("#result").textContent = "요청 중...";

    try {
        // 4. 서버에 GET 요청
        const res = await fetch(url, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        // 5. 응답 파싱
        const data = await res.json();

        // 6. 에러 처리 (상태 코드 확인 및 서버 반환 ok 값 확인)
        if (!res.ok || data.ok === false) {
            $("#result").innerHTML = `<span class='error'>에러 : ${data.error}</span>`;
            return;
        }

        // 7. 성공 시 화면 출력
        $("#result").innerHTML = `
            <div>이름 : ${data.name}</div>
            <div>나이 : ${data.age}</div>
            <div>연령대 : ${data.age_group}</div>
            <div>메세지 : ${data.message}</div>
        `;

    } catch (err) {
        // 네트워크 장애 등 catch 블록 처리
        $("#result").innerHTML = `<span class='error'>네트워크 또는 파싱 오류 : ${err}</span>`;
    }
});