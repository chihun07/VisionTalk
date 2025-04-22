import cv2
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os
import json
import numpy as np
import time
import mediapipe as mp
import threading
import tkinter as tk
import speech_recognition as sr

# -------------------------------------------------------------
# 1. 초기 설정 및 라이브러리 임포트
# -------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 전역 변수 (영상 분석 결과 및 질문 답변을 저장)
latest_analysis_result = {"result": None}
recording_flag = False
recorded_question = ""
recording_thread = None

# -------------------------------------------------------------
# 2. 보조 함수들
# -------------------------------------------------------------
def clean_json_text(text):
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text

def analyze_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    prompt = (
        "아래 이미지를 분석하여, 인물의 속성을 다음 JSON 형식으로 출력해줘:\n"
        "{\n"
        '  "성별": "",\n'
        '  "얼굴": "",\n'
        '  "표정": "",\n'
        '  "감정": "",\n'
        '  "입고 있는 옷": "",\n'
        '  "손이 어떤 모양인지": "",\n'
        '  "손에 들고 있는것": "" or "없음"\n'
        "}\n"
        "분석 결과는 한국어로 작성해줘."
    )
    
    response = model.generate_content([prompt, pil_image])
    raw_text = response.text
    cleaned_text = clean_json_text(raw_text)
    
    try:
        result_json = json.loads(cleaned_text)
        # 질문에 대한 답변 필드를 추가 (초기에는 빈 문자열)
        result_json["질문의 대답"] = ""
    except Exception as e:
        result_json = None
    return result_json, cleaned_text

def draw_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    draw = ImageDraw.Draw(pil_img)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def get_hand_bounding_box(image, hand_landmarks):
    img_h, img_w, _ = image.shape
    x_coords = [lm.x * img_w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * img_h for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def get_object_bounding_box(hand_box, margin=10):
    x, y, w, h = hand_box
    obj_x = max(0, x - margin)
    obj_y = max(0, y - int(h * 0.8) - margin)
    obj_w = w + (margin * 2)
    obj_h = int(h * 0.8) + margin
    return (obj_x, obj_y, obj_w, obj_h)

def analysis_worker(frame_copy, result_container):
    result_json, cleaned_text = analyze_frame(frame_copy)
    result_container["result"] = result_json
    if result_json is not None:
        print("\nGemini Vision 분석 결과 (원본 텍스트):")
        print(cleaned_text)
        print("\n분석 결과 (JSON):")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print("분석 결과가 output.json 파일에 저장되었습니다.")
    else:
        result_container["result"] = None

def answer_question(question_text):
    prompt = f"다음 질문에 대답해줘:\n{question_text}\n답은 한국어로 작성해줘."
    response = model.generate_content([prompt])
    answer = response.text.strip()
    answer = clean_json_text(answer)
    return answer

# -------------------------------------------------------------
# 3. 마이크 입력 관련 함수 (버튼을 누르고 있는 동안 연속 녹음)
# -------------------------------------------------------------
def continuous_record():
    global recorded_question, recording_flag
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while recording_flag:
            try:
                # 짧은 구간(예: 2초) 동안 녹음
                audio = r.listen(source, phrase_time_limit=2)
                try:
                    text = r.recognize_google(audio, language="ko-KR")
                    recorded_question += " " + text
                except Exception as e:
                    print("인식 중 오류:", e)
            except Exception as e:
                print("녹음 중 오류:", e)

def on_press(event, text_widget):
    global recording_flag, recording_thread, recorded_question
    recording_flag = True
    recorded_question = ""
    text_widget.insert(tk.END, "녹음 시작...\n")
    recording_thread = threading.Thread(target=continuous_record)
    recording_thread.start()

def on_release(event, text_widget):
    global recording_flag, recording_thread, recorded_question, latest_analysis_result
    recording_flag = False
    if recording_thread is not None:
        recording_thread.join()
    question = recorded_question.strip()
    text_widget.insert(tk.END, f"질문: {question}\n")
    if question:
        answer = answer_question(question)
        text_widget.insert(tk.END, f"답변: {answer}\n")
        # 영상 분석 결과 JSON에 질문 답변 업데이트
        if latest_analysis_result["result"] is not None:
            latest_analysis_result["result"]["질문의 대답"] = answer
            with open("output.json", "w", encoding="utf-8") as f:
                json.dump(latest_analysis_result["result"], f, ensure_ascii=False, indent=2)
    else:
        text_widget.insert(tk.END, "질문 인식 실패\n")

# -------------------------------------------------------------
# 4. 메인 함수: tkinter 내에 카메라 영상과 질문하기 기능 통합
# -------------------------------------------------------------
def main():
    global latest_analysis_result
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    root = tk.Tk()
    root.title("실시간 영상과 질문하기")
    
    # 영상 출력용 라벨
    video_label = tk.Label(root)
    video_label.pack(padx=10, pady=10)
    
    # 질문/답변 결과를 보여줄 텍스트 위젯
    text_widget = tk.Text(root, height=10, width=50, font=("Arial", 12))
    text_widget.pack(padx=10, pady=10)
    
    # "질문하기 (누르고 있는 동안 녹음)" 버튼: 마우스 버튼 이벤트로 녹음 시작/종료 처리
    question_button = tk.Button(root, text="질문하기 (누르고 있는 동안 녹음)", font=("Arial", 14))
    question_button.pack(padx=10, pady=10)
    question_button.bind("<ButtonPress-1>", lambda event: on_press(event, text_widget))
    question_button.bind("<ButtonRelease-1>", lambda event: on_release(event, text_widget))
    
    analysis_interval = 5
    last_analysis_time = time.time()
    analysis_thread = None

    def update_frame():
        nonlocal last_analysis_time, analysis_thread
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            root.after(10, update_frame)
            return
        
        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        
        # 얼굴 검출 및 오버레이
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if latest_analysis_result["result"] is not None:
                gender = latest_analysis_result["result"].get("성별", "")
                expression = latest_analysis_result["result"].get("표정", "")
                gender_label = f"성별: {gender}"
                expression_label = f"표정: {expression}"
                frame = draw_korean_text(frame, gender_label, (x, y - 60), font_size=20, color=(0, 255, 0))
                frame = draw_korean_text(frame, expression_label, (x, y - 30), font_size=20, color=(0, 255, 0))
        
        # 손 검출 및 오버레이
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands_detector.process(rgb_frame)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_box = get_hand_bounding_box(frame, hand_landmarks)
                bx, by, bw, bh = hand_box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
                if latest_analysis_result["result"] is not None:
                    hand_item = latest_analysis_result["result"].get("손에 들고 있는것", "").strip()
                    hand_shape = latest_analysis_result["result"].get("손이 어떤 모양인지", "").strip()
                    if hand_item and hand_item != "없음":
                        obj_box = get_object_bounding_box(hand_box)
                        ox, oy, ow, oh = obj_box
                        cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 255, 255), 2)
                        label = f"물체: {hand_item}"
                        frame = draw_korean_text(frame, label, (ox, oy - 30), font_size=20, color=(0, 255, 255))
                    else:
                        label = f"손 모양: {hand_shape}"
                        frame = draw_korean_text(frame, label, (bx, by - 30), font_size=20, color=(255, 0, 0))
        
        # 분석 주기 처리 (예: 5초마다)
        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval and (analysis_thread is None or not analysis_thread.is_alive()):
            last_analysis_time = current_time
            frame_copy = frame.copy()
            analysis_thread = threading.Thread(target=analysis_worker, args=(frame_copy, latest_analysis_result))
            analysis_thread.start()
        
        # tkinter에 표시할 수 있도록 이미지 변환 (BGR -> RGB -> PhotoImage)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        video_label.imgtk = imgtk  # 이미지 참조 유지
        video_label.configure(image=imgtk)
        
        root.after(10, update_frame)
    
    update_frame()
    root.mainloop()
    cap.release()

if __name__ == "__main__":
    main()
