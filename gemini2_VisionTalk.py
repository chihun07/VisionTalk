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
print(f"API 키: {API_KEY}")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 전역 변수 (영상 분석 결과 및 질문 답변을 저장)
latest_analysis_result = {"result": None}

# 마이크 사용 가능 여부
microphone_available = True
try:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        pass
except Exception as e:
    print("마이크 초기화 실패:", e)
    print("음성 인식 기능이 비활성화됩니다.")
    microphone_available = False

# (listen_in_background) 종료 함수 포인터
stop_listening = None

# (중간 인식 결과도 확인할 수 있도록) 전역 변수
voice_recorded_text = ""

# 최신 카메라 프레임 저장 (텍스트 질문 시 이미지 전달용)
latest_camera_frame = None

# API 쿨다운 관련 변수
api_cooldown = False
api_cooldown_start = 0
API_COOLDOWN_DURATION = 30  # 30초 쿨다운

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

def check_api_available():
    global api_cooldown, api_cooldown_start
    if api_cooldown:
        current_time = time.time()
        if current_time - api_cooldown_start >= API_COOLDOWN_DURATION:
            api_cooldown = False
            return True
        return False
    return True

def start_api_cooldown():
    global api_cooldown, api_cooldown_start
    api_cooldown = True
    api_cooldown_start = time.time()
    print(f"API 사용량 초과로 {API_COOLDOWN_DURATION}초 동안 쿨다운됩니다.")

def analyze_frame(frame):
    if not check_api_available():
        remaining_time = int(API_COOLDOWN_DURATION - (time.time() - api_cooldown_start))
        print(f"API 쿨다운 중: {remaining_time}초 남음")
        return None, f"API 쿨다운 중: {remaining_time}초 남음"
    
    h, w = frame.shape[:2]
    new_width = 640
    new_height = int(h * (new_width / w))
    small_frame = cv2.resize(frame, (new_width, new_height))
    
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
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
    
    try:
        response = model.generate_content([prompt, pil_image])
        raw_text = response.text
        cleaned_text = clean_json_text(raw_text)
        
        try:
            result_json = json.loads(cleaned_text)
            result_json["질문의 대답"] = ""
        except Exception:
            result_json = None
        return result_json, cleaned_text
    except Exception as e:
        if "quota" in str(e).lower():
            start_api_cooldown()
            return None, "API 사용량 초과로 쿨다운 시작"
        return None, str(e)

def draw_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    draw = ImageDraw.Draw(pil_img)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
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

def answer_question(question_text, frame):
    if not check_api_available():
        remaining_time = int(API_COOLDOWN_DURATION - (time.time() - api_cooldown_start))
        return f"API 쿨다운 중: {remaining_time}초 남음"
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt = f"다음 질문에 대답해줘:\n{question_text}\n답은 한국어로 작성해줘."
        response = model.generate_content([prompt, pil_image])
        answer = response.text.strip()
        return clean_json_text(answer)
    except Exception as e:
        if "quota" in str(e).lower():
            start_api_cooldown()
            return "API 사용량 초과로 쿨다운 시작"
        return str(e)

def answer_text_question_with_image(question_text, frame):
    if not check_api_available():
        remaining_time = int(API_COOLDOWN_DURATION - (time.time() - api_cooldown_start))
        return f"API 쿨다운 중: {remaining_time}초 남음"
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt = f"다음 질문에 대답해줘:\n{question_text}\n답은 한국어로 작성해줘."
        response = model.generate_content([prompt, pil_image])
        answer = response.text.strip()
        return clean_json_text(answer)
    except Exception as e:
        if "quota" in str(e).lower():
            start_api_cooldown()
            return "API 사용량 초과로 쿨다운 시작"
        return str(e)

# -------------------------------------------------------------
# 음성 인식 콜백: 중간 인식 결과도 즉시 표시
# -------------------------------------------------------------
def speech_callback(recognizer, audio):
    global voice_recorded_text
    if not microphone_available:
        return
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        voice_recorded_text += " " + text
        # 메인 스레드에서 UI 업데이트
        root.after(0, lambda: voice_input_box.insert(tk.END, f"중간 인식: {text}\n"))
        root.after(0, lambda: voice_input_box.see(tk.END))
    except Exception as e:
        print("음성 인식 오류:", e)

# -------------------------------------------------------------
# 음성 입력 토글 (별도 스레드에서 최종 결과 처리)
# -------------------------------------------------------------
def toggle_voice_input():
    global stop_listening, voice_recorded_text, latest_camera_frame
    if not microphone_available:
        tk.messagebox.showwarning("경고", "마이크를 사용할 수 없습니다.")
        return

    if stop_listening is None:
        mic = sr.Microphone()
        stop_listening = r.listen_in_background(mic, speech_callback, phrase_time_limit=3)
        voice_input_button.config(text="녹음 중지")
        voice_recorded_text = ""
        voice_input_box.delete("1.0", tk.END)
        voice_input_box.insert(tk.END, "녹음을 시작했습니다.\n")
    else:
        stop_listening(wait_for_stop=False)
        stop_listening = None
        voice_input_button.config(text="음성으로 입력")
        # 별도 스레드에서 최종 음성 인식 결과 처리
        threading.Thread(target=process_voice_input).start()

def process_voice_input():
    global voice_recorded_text, latest_camera_frame, latest_analysis_result
    if voice_recorded_text.strip():
        root.after(0, lambda: voice_input_box.insert(tk.END, "\n=== 최종 인식 결과 ===\n"))
        root.after(0, lambda: voice_input_box.insert(tk.END, f"{voice_recorded_text.strip()}\n"))
        
        answer = answer_question(voice_recorded_text.strip(), latest_camera_frame)
        root.after(0, lambda: voice_input_box.insert(tk.END, f"답변: {answer}\n"))
        
        if latest_analysis_result["result"] is not None:
            latest_analysis_result["result"]["질문의 대답"] = answer
            with open("output.json", "w", encoding="utf-8") as f:
                json.dump(latest_analysis_result["result"], f, ensure_ascii=False, indent=2)
    else:
        root.after(0, lambda: voice_input_box.insert(tk.END, "질문 인식 실패\n"))

# -------------------------------------------------------------
# 텍스트 입력 관련 함수 (별도 스레드에서 실행)
# -------------------------------------------------------------
def handle_text_question():
    question = text_input_box.get("1.0", tk.END).strip()
    text_input_box.delete("1.0", tk.END)
    if not question:
        root.after(0, lambda: text_input_box.insert(tk.END, "경고: 질문을 입력해주세요.\n"))
        return
    if latest_camera_frame is None:
        root.after(0, lambda: text_input_box.insert(tk.END, "오류: 현재 카메라 이미지를 가져올 수 없습니다.\n"))
        return
    answer = answer_text_question_with_image(question, latest_camera_frame)
    root.after(0, lambda: text_input_box.insert(tk.END, f"질문: {question}\n답변: {answer}\n"))

def on_text_question():
    threading.Thread(target=handle_text_question).start()

# -------------------------------------------------------------
# 분석 결과 텍스트 박스 주기 업데이트 함수
# -------------------------------------------------------------
def update_analysis_text():
    analysis_result_box.config(state="normal")
    analysis_result_box.delete("1.0", tk.END)
    
    if api_cooldown:
        remaining_time = int(API_COOLDOWN_DURATION - (time.time() - api_cooldown_start))
        analysis_result_box.insert(tk.END, f"API 쿨다운 중: {remaining_time}초 남음\n\n")
    
    if latest_analysis_result["result"] is not None:
        pretty_text = json.dumps(latest_analysis_result["result"], indent=2, ensure_ascii=False)
        analysis_result_box.insert(tk.END, pretty_text)
    
    analysis_result_box.config(state="disabled")
    root.after(1000, update_analysis_text)

# -------------------------------------------------------------
# 메인 함수
# -------------------------------------------------------------
def main():
    global latest_camera_frame, root
    global text_input_box, voice_input_box, analysis_result_box, voice_input_button
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    display_width = 1280
    display_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    root = tk.Tk()
    root.title("실시간 영상과 질문하기")
    
    # 상단: 카메라 영상
    video_label = tk.Label(root)
    video_label.pack(padx=10, pady=10)
    
    # 하단 프레임
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # (1) 텍스트 입력 박스
    text_input_frame = tk.Frame(bottom_frame)
    text_input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    text_input_box = tk.Text(text_input_frame, height=10, width=40, font=("Arial", 12))
    text_input_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    text_input_button = tk.Button(text_input_frame, text="텍스트로 입력받기", font=("Arial", 14), command=on_text_question)
    text_input_button.pack(side=tk.TOP, pady=5)
    
    # (2) 음성 입력 박스
    voice_input_frame = tk.Frame(bottom_frame)
    voice_input_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    voice_input_box = tk.Text(voice_input_frame, height=10, width=40, font=("Arial", 12))
    voice_input_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    voice_input_button = tk.Button(voice_input_frame, text="음성으로 입력", font=("Arial", 14), command=toggle_voice_input)
    voice_input_button.pack(side=tk.TOP, pady=5)
    
    # (3) 분석 결과 출력 박스
    analysis_result_frame = tk.Frame(bottom_frame)
    analysis_result_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
    analysis_result_box = tk.Text(analysis_result_frame, height=10, width=40, font=("Arial", 12))
    analysis_result_box.pack(fill=tk.BOTH, expand=True)
    analysis_result_box.config(state="disabled")
    
    bottom_frame.grid_columnconfigure(0, weight=1)
    bottom_frame.grid_columnconfigure(1, weight=1)
    bottom_frame.grid_columnconfigure(2, weight=1)
    
    analysis_mode_var = tk.BooleanVar(value=True)
    analysis_mode_checkbox = tk.Checkbutton(root, text="분석 모드", variable=analysis_mode_var, font=("Arial", 12))
    analysis_mode_checkbox.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)
    
    analysis_interval = 5
    last_analysis_time = time.time()
    analysis_thread = None

    def update_frame():
        nonlocal last_analysis_time, analysis_thread
        global latest_camera_frame
        
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            root.after(10, update_frame)
            return
        
        frame = cv2.flip(frame, 1)
        latest_camera_frame = frame.copy()
        
        if analysis_mode_var.get():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if latest_analysis_result["result"] is not None:
                    gender = latest_analysis_result["result"].get("성별", "")
                    expression = latest_analysis_result["result"].get("표정", "")
                    gender_label = f"성별: {gender}"
                    expression_label = f"표정: {expression}"
                    frame = draw_korean_text(frame, gender_label, (x, y - 60), font_size=20, color=(0, 255, 0))
                    frame = draw_korean_text(frame, expression_label, (x, y - 30), font_size=20, color=(0, 255, 0))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hands_detector.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    hand_box = get_hand_bounding_box(frame, hand_landmarks)
                    bx, by, bw, bh = hand_box
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                    if latest_analysis_result["result"] is not None:
                        hand_item = latest_analysis_result["result"].get("손에 들고 있는것", "").strip()
                        hand_shape = latest_analysis_result["result"].get("손이 어떤 모양인지", "").strip()
                        if hand_item and hand_item != "없음":
                            obj_box = get_object_bounding_box(hand_box)
                            ox, oy, ow, oh = obj_box
                            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 2)
                            label = f"물체: {hand_item}"
                            frame = draw_korean_text(frame, label, (ox, oy - 30), font_size=20, color=(0, 255, 255))
                        else:
                            label = f"손 모양: {hand_shape}"
                            frame = draw_korean_text(frame, label, (bx, by - 30), font_size=20, color=(255, 0, 0))
        
        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval and (analysis_thread is None or not analysis_thread.is_alive()):
            if check_api_available():
                last_analysis_time = current_time
                frame_copy = frame.copy()
                analysis_thread = threading.Thread(target=analysis_worker, args=(frame_copy, latest_analysis_result))
                analysis_thread.start()
                print("이미지 분석 시작...")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        root.after(10, update_frame)
    
    update_frame()
    update_analysis_text()
    root.mainloop()
    cap.release()

if __name__ == "__main__":
    main()
