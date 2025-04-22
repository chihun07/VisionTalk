

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
import sounddevice as sd
import absl.logging
import queue
import concurrent.futures  # 스레드 풀 사용

# -------------------------------------------------------------
# 설정 값 (클래스 내부에 정의될 수도 있음)
# -------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
MODEL_NAME = 'gemini-1.5-flash'
API_COOLDOWN_DURATION = 30
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 폰트 경로는 환경에 맞게 조정
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000

# -------------------------------------------------------------
# Gemini API 핸들러 클래스
# -------------------------------------------------------------
class GeminiAPIHandler:
    def __init__(self, api_key, model_name):
        """Gemini API를 사용하기 위한 핸들러 클래스"""
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.configure_api()
        self.cooldown = False
        self.cooldown_start = 0

    def configure_api(self):
        """Gemini API를 설정합니다."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print("Gemini API 설정 완료.")
        except Exception as e:
            print(f"Gemini API 설정 실패: {e}")

    def check_api_available(self):
        """API 사용량 제한을 확인하고 쿨다운 여부를 반환합니다."""
        if self.cooldown:
            current_time = time.time()
            if current_time - self.cooldown_start >= API_COOLDOWN_DURATION:
                self.cooldown = False
                return True
            return False
        return True

    def start_api_cooldown(self):
        """API 쿨다운을 시작합니다."""
        self.cooldown = True
        self.cooldown_start = time.time()
        print(f"API 사용량 초과로 {API_COOLDOWN_DURATION}초 동안 쿨다운됩니다.")

    def analyze_frame(self, frame):
        """프레임을 분석하고, 인물의 속성을 JSON 형태로 반환합니다."""
        if not self.check_api_available():
            remaining_time = int(API_COOLDOWN_DURATION - (time.time() - self.cooldown_start))
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
            response = self.model.generate_content([prompt, pil_image])
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
                self.start_api_cooldown()
                return None, "API 사용량 초과로 쿨다운 시작"
            return None, str(e)

    def answer_question(self, question_text, frame):
        """질문과 이미지를 Gemini에게 전달하여 답변을 받습니다."""
        if not self.check_api_available():
            remaining_time = int(API_COOLDOWN_DURATION - (time.time() - self.cooldown_start))
            return f"API 쿨다운 중: {remaining_time}초 남음"

        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prompt = f"다음 질문에 대답해줘:\n{question_text}\n답은 한국어로 작성해줘."
            response = self.model.generate_content([prompt, pil_image])
            answer = response.text.strip()
            return clean_json_text(answer)
        except Exception as e:
            if "quota" in str(e).lower():
                self.start_api_cooldown()
                return "API 사용량 초과로 쿨다운 시작"
            return str(e)

    def answer_wav(self, question_text, wav_file_path):
        """WAV 파일을 Gemini에게 전달하여 텍스트 답변을 받습니다."""
        if not self.check_api_available():
            remaining_time = int(API_COOLDOWN_DURATION - (time.time() - self.cooldown_start))
            return f"API 쿨다운 중: {remaining_time}초 남음"

        try:
            with open(wav_file_path, "rb") as f:
                wav_data = f.read()

            # Gemini API가 오디오 파일을 직접 지원하는지 확인 필요
            # 현재는 텍스트로 변환된 음성 데이터로 질문
            prompt = f"음성 질문(파일 {wav_file_path})입니다.\n질문: {question_text}\n답은 한국어로 작성해줘."
            response = self.model.generate_content([prompt, wav_data])  # 수정 필요!
            answer = response.text.strip()
            return clean_json_text(answer)
        except FileNotFoundError:
            return f"오류: 파일 {wav_file_path}을 찾을 수 없습니다."
        except Exception as e:
            if "quota" in str(e).lower():
                self.start_api_cooldown()
                return "API 사용량 초과로 쿨다운 시작"
            return str(e)

# -------------------------------------------------------------
# 카메라 핸들러 클래스
# -------------------------------------------------------------
class CameraHandler:
    def __init__(self, camera_index=0, display_width=1280, display_height=720):
        """웹캠을 제어하고 프레임을 캡처하는 클래스"""
        self.camera_index = camera_index
        self.display_width = display_width
        self.display_height = display_height
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def initialize(self):
        """웹캠을 초기화합니다."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise ValueError("카메라를 열 수 없습니다.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            print("카메라 초기화 완료.")
        except Exception as e:
            print(f"카메라 초기화 실패: {e}")
            return False
        return True

    def read_frame(self):
        """웹캠에서 프레임을 읽어옵니다."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("프레임을 가져올 수 없습니다.")
            return cv2.flip(frame, 1)  # 좌우 반전
        except Exception as e:
            print(f"프레임 읽기 오류: {e}")
            return None

    def detect_faces(self, frame):
        """프레임에서 얼굴을 감지합니다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def release(self):
        """웹캠을 해제합니다."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("카메라 해제 완료.")

# -------------------------------------------------------------
# 음성 입력 핸들러 클래스
# -------------------------------------------------------------
class VoiceInputHandler:
    def __init__(self, chunk_size=CHUNK_SIZE, sample_rate=SAMPLE_RATE):
        """음성 입력을 처리하는 클래스"""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.stop_listening = None
        self.voice_recorded_text = ""
        self.chunk_index = 0
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

    def initialize_microphone(self):
        """마이크를 초기화합니다."""
        try:
            self.microphone = SoundDeviceMicrophone(chunk_size=self.chunk_size, sample_rate=self.sample_rate)
            with self.microphone as source:
                pass  # 마이크가 사용 가능한지 테스트
            print("마이크 초기화 완료.")
            return True
        except Exception as e:
            print(f"마이크 초기화 실패: {e}")
            print("음성 인식 기능이 비활성화됩니다.")
            return False

    def speech_callback(self, recognizer, audio):
        """음성 인식 콜백 함수: 중간 인식 결과 처리 및 WAV 파일 저장"""
        try:
            # 1) 먼저 현재 chunk를 WAV 형태로 저장
            chunk_filename = f"chunk_{self.chunk_index}.wav"
            self.chunk_index += 1
            with open(chunk_filename, "wb") as f:
                f.write(audio.get_wav_data())

            # 2) 이어서 구글 STT로 해당 chunk에 대한 인식 결과를 추출
            try:
                text = recognizer.recognize_google(audio, language="ko-KR")
                self.voice_recorded_text += " " + text
                # 메인 스레드에서 UI 업데이트
                root.after(0, lambda: voice_input_box.insert(tk.END, f"중간 인식: {text}\n"))
                root.after(0, lambda: voice_input_box.see(tk.END))
            except Exception as e:
                print("음성 인식 오류:", e)
        except Exception as e:
            print(f"콜백 함수 오류: {e}")

    def toggle_voice_input(self, gemini_handler, latest_camera_frame):
        """음성 입력 기능을 켜고 끄는 함수"""
        if self.stop_listening is None:
            mic = self.microphone
            # phrase_time_limit=3 → 3초간 말이 없으면 문장 단위 종료 처리
            self.stop_listening = self.recognizer.listen_in_background(mic, self.speech_callback, phrase_time_limit=3)
            voice_input_button.config(text="녹음 중지")
            self.voice_recorded_text = ""
            voice_input_box.delete("1.0", tk.END)
            voice_input_box.insert(tk.END, "녹음을 시작했습니다.\n")
        else:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None
            voice_input_button.config(text="음성으로 입력")
            # 별도 스레드에서 최종 음성 인식 결과 처리
            threading.Thread(target=self.process_voice_input, args=(gemini_handler, latest_camera_frame)).start()

    def process_voice_input(self, gemini_handler, latest_camera_frame):
        """최종 음성 인식 결과를 처리하고 Gemini에게 질문합니다."""
        if self.voice_recorded_text.strip():
            root.after(0, lambda: voice_input_box.insert(tk.END, "\n=== 최종 인식 결과 ===\n"))
            root.after(0, lambda: voice_input_box.insert(tk.END, f"{self.voice_recorded_text.strip()}\n"))

            answer = gemini_handler.answer_question(self.voice_recorded_text.strip(), latest_camera_frame)
            root.after(0, lambda: voice_input_box.insert(tk.END, f"답변: {answer}\n"))

            # 분석 결과 업데이트 (GUIHandler로 이동 고려)
            if latest_analysis_result["result"] is not None:
                latest_analysis_result["result"]["질문의 대답"] = answer
                with open("output.json", "w", encoding="utf-8") as f:
                    json.dump(latest_analysis_result["result"], f, ensure_ascii=False, indent=2)
        else:
            root.after(0, lambda: voice_input_box.insert(tk.END, "질문 인식 실패\n"))

# -------------------------------------------------------------
# GUI 핸들러 클래스
# -------------------------------------------------------------
class GUIHandler:
    def __init__(self, gemini_handler, camera_handler, voice_input_handler, window_title="실시간 영상과 질문하기"):
        """GUI를 관리하는 클래스"""
        self.root = tk.Tk()
        self.root.title(window_title)
        self.gemini_handler = gemini_handler
        self.camera_handler = camera_handler
        self.voice_input_handler = voice_input_handler
        self.latest_camera_frame = None
        self.analysis_mode_var = tk.BooleanVar(value=True)
        self.analysis_thread = None  # 분석 스레드
        self.last_analysis_time = 0
        self.analysis_interval = 5
        self.video_label = None
        self.text_input_box = None
        self.voice_input_box = None
        self.analysis_result_box = None
        self.voice_input_button = None
        self.text_input_button = None

        self.create_widgets()

    def create_widgets(self):
        """GUI 위젯을 생성합니다."""
        # 상단: 카메라 영상
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        # 하단 프레임
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # (1) 텍스트 입력 박스
        text_input_frame = tk.Frame(bottom_frame)
        text_input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.text_input_box = tk.Text(text_input_frame, height=10, width=40, font=("Arial", 12))
        self.text_input_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.text_input_button = tk.Button(text_input_frame, text="텍스트로 입력받기", font=("Arial", 14), command=self.on_text_question)
        self.text_input_button.pack(side=tk.TOP, pady=5)

        # (2) 음성 입력 박스
        voice_input_frame = tk.Frame(bottom_frame)
        voice_input_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.voice_input_box = tk.Text(voice_input_frame, height=10, width=40, font=("Arial", 12))
        self.voice_input_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.voice_input_button = tk.Button(voice_input_frame, text="음성으로 입력", font=("Arial", 14), command=self.toggle_voice_input)
        self.voice_input_button.pack(side=tk.TOP, pady=5)

        # (3) 분석 결과 출력 박스
        analysis_result_frame = tk.Frame(bottom_frame)
        analysis_result_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.analysis_result_box = tk.Text(analysis_result_frame, height=10, width=40, font=("Arial", 12))
        self.analysis_result_box.pack(fill=tk.BOTH, expand=True)
        self.analysis_result_box.config(state="disabled")

        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)

        analysis_mode_checkbox = tk.Checkbutton(self.root, text="분석 모드", variable=self.analysis_mode_var, font=("Arial", 12))
        analysis_mode_checkbox.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)

    def update_analysis_text(self):
        """분석 결과 텍스트 박스 주기 업데이트 함수"""
        self.analysis_result_box.config(state="normal")
        self.analysis_result_box.delete("1.0", tk.END)

        if self.gemini_handler.cooldown:
            remaining_time = int(API_COOLDOWN_DURATION - (time.time() - self.gemini_handler.cooldown_start))
            self.analysis_result_box.insert(tk.END, f"API 쿨다운 중: {remaining_time}초 남음\n\n")

        if latest_analysis_result["result"] is not None:
            pretty_text = json.dumps(latest_analysis_result["result"], indent=2, ensure_ascii=False)
            self.analysis_result_box.insert(tk.END, pretty_text)

        self.analysis_result_box.config(state="disabled")
        self.root.after(1000, self.update_analysis_text)  # 1초마다 갱신

    def draw_overlays(self, frame):
      """얼굴 감지 및 손 감지 결과를 프레임에 그립니다."""
      if self.analysis_mode_var.get():
          faces = self.camera_handler.detect_faces(frame)
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
      return frame

    def update_frame(self, executor):
        """웹캠에서 프레임을 읽어와 GUI에 표시하고, 주기적으로 분석을 수행합니다."""
        frame = self.camera_handler.read_frame()
        if frame is None:
            self.root.after(10, lambda: self.update_frame(executor))  # 재귀 호출
            return

        self.latest_camera_frame = frame.copy()

        frame = self.draw_overlays(frame)

        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval and (self.analysis_thread is None or self.analysis_thread.done()):
            if self.gemini_handler.check_api_available():
                self.last_analysis_time = current_time
                frame_copy = frame.copy()
                self.analysis_thread = executor.submit(analysis_worker, frame_copy, latest_analysis_result, self.gemini_handler)
                print("이미지 분석 시작...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, lambda: self.update_frame(executor))

    def on_text_question(self):
        """텍스트 질문을 처리합니다."""
        threading.Thread(target=self.handle_text_question).start()

    def handle_text_question(self):
        """텍스트 질문을 처리하고 Gemini에게 답변을 요청합니다."""
        question = self.text_input_box.get("1.0", tk.END).strip()
        self.text_input_box.delete("1.0", tk.END)
        if not question:
            root.after(0, lambda: self.text_input_box.insert(tk.END, "경고: 질문을 입력해주세요.\n"))
            return
        if self.latest_camera_frame is None:
            root.after(0, lambda: self.text_input_box.insert(tk.END, "오류: 현재 카메라 이미지를 가져올 수 없습니다.\n"))
            return
        answer = self.gemini_handler.answer_question(question, self.latest_camera_frame)
        root.after(0, lambda: self.text_input_box.insert(tk.END, f"질문: {question}\n답변: {answer}\n"))

    def toggle_voice_input(self):
        """음성 입력을 토글합니다."""
        if not microphone_available:
            tk.messagebox.showwarning("경고", "마이크를 사용할 수 없습니다.")
            return
        self.voice_input_handler.toggle_voice_input(self.gemini_handler, self.latest_camera_frame)

    def run(self, executor):
        """GUI를 실행합니다."""
        self.update_frame(executor)
        self.update_analysis_text()
        self.root.mainloop()

# -------------------------------------------------------------
# 메인 함수
# -------------------------------------------------------------
def main():
    """프로그램의 메인 함수"""
    absl.logging.set_verbosity(absl.logging.ERROR)

    # 핸들러 객체 생성
    gemini_handler = GeminiAPIHandler(API_KEY, MODEL_NAME)
    camera_handler = CameraHandler()
    voice_input_handler = VoiceInputHandler()

    # 초기화
    if not camera_handler.initialize():
        print("카메라 초기화 실패. 프로그램을 종료합니다.")
        return
    microphone_available = voice_input_handler.initialize_microphone()
    if not microphone_available:
        print("마이크 초기화 실패. 음성 입력 기능은 비활성화됩니다.")

    # GUI 핸들러 생성 및 실행
    gui_handler = GUIHandler(gemini_handler, camera_handler, voice_input_handler)

    # 스레드 풀 생성
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        gui_handler.run(executor)

    # 종료
    camera_handler.release()

if __name__ == "__main__":
    main()
