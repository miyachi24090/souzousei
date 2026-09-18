# 1. 必要なライブラリのインストール
!pip install face_recognition

import base64
import io
import time
import cv2
import face_recognition
import numpy as np
from PIL import Image
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js

# 2. Google Colab上でリアルタイムWebカメラストリームを管理するJavaScript関数の定義
def video_stream():
    """
    ブラウザのWebカメラを起動し、ビデオ映像と透明オーバーレイ用のキャンバスを画面に配置する。
    """
    js = Javascript('''
        var video;
        var div = null;
        var stream;
        var captureCanvas;
        var imgElement;
        var pendingResolve = null;
        var shutdown = false;

        function removeDom() {
            if (stream) {
                stream.getVideoTracks()[0].stop();
            }
            if (video) video.remove();
            if (div) div.remove();
            video = null;
            div = null;
            stream = null;
            imgElement = null;
            captureCanvas = null;
        }

        function onAnimationFrame() {
            if (!shutdown) {
                window.requestAnimationFrame(onAnimationFrame);
            }
            if (pendingResolve) {
                var result = "";
                if (!shutdown && captureCanvas && video) {
                    captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                    result = captureCanvas.toDataURL('image/jpeg', 0.8);
                }
                var lp = pendingResolve;
                pendingResolve = null;
                lp(result);
            }
        }

        async function createDom() {
            if (div !== null) {
                return stream;
            }

            div = document.createElement('div');
            div.style.border = '2px solid #4285F4';
            div.style.padding = '8px';
            div.style.width = 'max-content';
            div.style.borderRadius = '8px';
            div.style.backgroundColor = '#1e1e1e';
            div.style.color = '#FFFFFF';
            div.style.fontFamily = 'Arial, sans-serif';

            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.style.marginBottom = '8px';

            const title = document.createElement('span');
            title.innerHTML = '⚡ <b>リアルタイム眠気検知ストリーム (EAR)</b>';
            title.style.fontSize = '14px';
            header.appendChild(title);

            const stopBtn = document.createElement('button');
            stopBtn.textContent = '⏹ 停止 (Stop)';
            stopBtn.style.padding = '4px 12px';
            stopBtn.style.fontSize = '12px';
            stopBtn.style.backgroundColor = '#EA4335';
            stopBtn.style.color = '#FFFFFF';
            stopBtn.style.border = 'none';
            stopBtn.style.borderRadius = '4px';
            stopBtn.style.cursor = 'pointer';
            stopBtn.onclick = () => { shutdown = true; };
            header.appendChild(stopBtn);

            div.appendChild(header);

            const streamContainer = document.createElement('div');
            streamContainer.style.position = 'relative';
            streamContainer.style.width = '640px';
            streamContainer.style.height = '480px';
            streamContainer.style.borderRadius = '6px';
            streamContainer.style.overflow = 'hidden';
            div.appendChild(streamContainer);

            video = document.createElement('video');
            video.style.display = 'block';
            video.width = 640;
            video.height = 480;
            video.setAttribute('playsinline', '');
            video.style.position = 'absolute';
            video.style.left = '0';
            video.style.top = '0';

            stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
            video.srcObject = stream;
            await video.play();

            streamContainer.appendChild(video);

            // Python側から返される解析結果オーバーレイ用の画像要素
            imgElement = document.createElement('img');
            imgElement.style.position = 'absolute';
            imgElement.style.left = '0';
            imgElement.style.top = '0';
            imgElement.style.width = '640px';
            imgElement.style.height = '480px';
            imgElement.style.pointerEvents = 'none';
            streamContainer.appendChild(imgElement);

            captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640;
            captureCanvas.height = 480;

            document.body.appendChild(div);
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            window.requestAnimationFrame(onAnimationFrame);
            return stream;
        }

        async function stream_frame(label, imgData) {
            if (shutdown) {
                removeDom();
                return "";
            }
            if (div === null) {
                await createDom();
            }
            if (imgData && imgElement) {
                imgElement.src = imgData;
            }
            return new Promise((resolve) => {
                pendingResolve = resolve;
            });
        }
    ''')
    display(js)

def video_frame(label, overlay_bytes):
    """
    JavaScriptと非同期通信し、最新フレーム（Base64 JPEG）を受信しつつ、
    作成したオーバーレイ画像（Base64 PNG）をブラウザ側に渡す。
    """
    data = eval_js(f'stream_frame("{label}", "{overlay_bytes}")')
    return data

# 3. Base64 ↔ OpenCV / NumPy 画像変換ユーティリティ
def js_to_image(js_reply):
    """
    Base64エンコードされたJPEG文字列をOpenCV画像（BGR NumPy配列）にデコード
    """
    header, encoded = js_reply.split(',', 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return img

def overlay_to_bytes(overlay_img):
    """
    RGBA形式（4チャンネル）のオーバーレイ画像をBase64 PNG形式にエンコード
    """
    overlay_pil = Image.fromarray(overlay_img, mode='RGBA')
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    overlay_bytes = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
    return overlay_bytes

# 4. 目の「開き具合（EAR: Eye Aspect Ratio）」を計算する関数
def calculate_ear(eye_landmarks):
    """
    目の周りの6点座標からEARを計算する。
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    # まぶたの上下の距離（縦の長さ）を計算
    p2_p6 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    p3_p5 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    # 目頭と目尻の距離（横の長さ）を計算
    p1_p4 = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))

    # ゼロ除算防止
    if p1_p4 == 0:
        return 0.0
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

# 5. リアルタイム・ストリーミング処理メインルーチン
print("Webカメラストリームを起動しています...")
video_stream()

# 初期オーバーレイ（完全透明なRGBA画像: 640x480）
empty_overlay = np.zeros((480, 640, 4), dtype=np.uint8)
overlay_bytes = overlay_to_bytes(empty_overlay)

# パラメータ設定
EAR_THRESHOLD = 0.22             # 眠気判定の閾値（0.20〜0.25が目安）
CONSECUTIVE_FRAMES_ALERT = 5     # 連続何フレーム閾値を下回ったら警告を出すか（まばたき誤検知対策）
sleepy_frame_count = 0           # 連続閉眼フレーム数カウンタ
prev_time = time.time()          # FPS計算用タイマー

try:
    while True:
        # JavaScriptから最新フレームを取得し、前回作成したオーバーレイを送信
        js_reply = video_frame("EAR Realtime Stream", overlay_bytes)
        if not js_reply:
            print("ストリーミングが停止されました。")
            break

        # フレーム画像をデコード
        frame_bgr = js_to_image(js_reply)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # 新しい透明オーバーレイレイヤーを作成 (Height: 480, Width: 640, Channel: 4 [RGBA])
        overlay = np.zeros((480, 640, 4), dtype=np.uint8)

        # AIで顔のパーツ（ランドマーク）を特定
        face_landmarks_list = face_recognition.face_landmarks(frame_rgb)

        if len(face_landmarks_list) == 0:
            sleepy_frame_count = 0
            # 顔が検出されない場合の案内テキスト（黄色: R=255, G=200, B=0, A=255）
            cv2.putText(overlay, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0, 255), 2, cv2.LINE_AA)
        else:
            landmarks = face_landmarks_list[0]

            # 左右の目の座標を取得
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']

            # 両目のEAR平均値を算出
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # 目の輪郭をシアン色で描画 (R=0, G=255, B=255, A=255)
            for eye in [left_eye, right_eye]:
                pts = np.array(eye, np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255, 255), thickness=2)

            # 眠気判定ロジック（時系列判定）
            if avg_ear < EAR_THRESHOLD:
                sleepy_frame_count += 1
            else:
                sleepy_frame_count = max(0, sleepy_frame_count - 1)

            # ステータス表示の分岐
            if sleepy_frame_count >= CONSECUTIVE_FRAMES_ALERT:
                status_text = f"!! DROWSINESS ALERT !! (EAR: {avg_ear:.2f})"
                status_color = (255, 0, 0, 255)  # 警告：赤色 (RGBA)
                # 警告枠の描画
                cv2.rectangle(overlay, (10, 10), (630, 470), (255, 0, 0, 255), 4)
            elif avg_ear < EAR_THRESHOLD:
                status_text = f"Status: BLINKING / LOW (EAR: {avg_ear:.2f})"
                status_color = (255, 165, 0, 255)  # まばたき/低下中：オレンジ色
            else:
                status_text = f"Status: AWAKE (EAR: {avg_ear:.2f})"
                status_color = (0, 255, 0, 255)  # 覚醒：緑色

            # ステータステキスト描画
            cv2.putText(overlay, status_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

            # EAR数値と連続閉眼カウントの表示
            info_text = f"Avg EAR: {avg_ear:.3f} | Sleepy Frames: {sleepy_frame_count}/{CONSECUTIVE_FRAMES_ALERT}"
            cv2.putText(overlay, info_text, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1, cv2.LINE_AA)

        # FPS情報の描画
        cv2.putText(overlay, f"FPS: {fps:.1f}", (520, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200, 255), 2, cv2.LINE_AA)

        # オーバーレイ画像をBase64文字列に変換して次ループで送信
        overlay_bytes = overlay_to_bytes(overlay)

except Exception as e:
    print(f"ストリーミング処理中にエラーが発生しました: {e}")