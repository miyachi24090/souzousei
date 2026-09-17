# 1. 必要なライブラリのインストールaaa
!pip install face_recognition

import base64
import io
import cv2
import face_recognition
import numpy as np
from PIL import Image
from IPython.display import display, Javascript
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow

# 2. Google Colab上でブラウザのWebカメラを起動して静止画をキャプチャする関数
def take_photo(filename='photo.jpg', quality=0.8):
    """
    JavaScript経由でブラウザのWebカメラストリームを起動し、
    キャプチャボタン押下で静止画をBase64形式でPython側に取得・保存する。
    """
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = '📸 キャプチャ（撮影）';
            capture.style.display = 'block';
            capture.style.margin = '10px 0';
            capture.style.padding = '8px 16px';
            capture.style.fontSize = '16px';
            capture.style.backgroundColor = '#4285F4';
            capture.style.color = '#FFFFFF';
            capture.style.border = 'none';
            capture.style.borderRadius = '4px';
            capture.style.cursor = 'pointer';

            const video = document.createElement('video');
            video.style.display = 'block';
            video.style.borderRadius = '8px';
            video.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
            
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            div.appendChild(capture);

            video.srcObject = stream;
            await video.play();

            // Google Colabの出力セルサイズにリサイズ
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // キャプチャボタンが押されるまで待機
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            // ストリーム停止とUI要素のクリーンアップ
            stream.getVideoTracks()[0].stop();
            div.remove();

            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    
    # Base64データをデコードしてバイナリに変換
    header, encoded = data.split(',', 1)
    binary = base64.b64decode(encoded)
    
    # ファイルとして保存
    with open(filename, 'wb') as f:
        f.write(binary)
    
    return filename

# 3. 目の「開き具合（EAR: Eye Aspect Ratio）」を計算する関数
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

    # 目の開き具合の比率（EAR）を算出（ゼロ除算防止）
    if p1_p4 == 0:
        return 0.0
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

# 4. Webカメラからの撮影実行と画像読み込み
print("Webカメラを起動します。「📸 キャプチャ」ボタンを押して撮影してください。")
try:
    image_file = take_photo('captured_face.jpg')
    print(f"画像が正常にキャプチャされました: {image_file}")
    
    # OpenCV / face_recognition 用に画像読み込み
    image_rgb = face_recognition.load_image_file(image_file)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 5. AIで顔のパーツ（ランドマーク）を特定する
    face_landmarks_list = face_recognition.face_landmarks(image_rgb)

    if len(face_landmarks_list) == 0:
        print("警告: 顔が検出されませんでした。正面を向いて再度撮影してください。")
    else:
        # 最初の1人分の顔データを処理
        landmarks = face_landmarks_list[0]

        # 左右の目の座標を取得
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        # 両目の開き具合の平均値を算出
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"目の開き具合（EAR値）: {avg_ear:.3f}")

        # 6. 眠気判定（基準値：0.22以下なら眠気ありと判定）
        THRESHOLD = 0.22

        if avg_ear < THRESHOLD:
            status_text = "Status: SLEEPY (Drowsy)"
            color = (0, 0, 255) # 赤色
        else:
            status_text = "Status: AWAKE (Active)"
            color = (0, 255, 0) # 緑色

    # 7. 画像に目の輪郭と判定結果を書き込む
    # 目の輪郭を線でなぞる
        for eye in [left_eye, right_eye]:
            pts = np.array(eye, np.int32)
        cv2.polylines(image_bgr, [pts], True, (255, 0, 0), 1)

    # 結果のテキストを画像の上に表示
    cv2.putText(image_bgr, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 画面に結果を表示
    cv2_imshow(image_bgr)

except Exception as err:
    print(f"エラーが発生しました: {err}")