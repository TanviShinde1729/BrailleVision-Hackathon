import os
import cv2
import numpy as np
import base64
import requests
import traceback
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD YOLO MODEL & DICTIONARY
# ---------------------------------------------------------
print("[INFO] Booting up YOLO Model...")
MODEL_PATH = "yolov8_braille.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")

CONFIDENCE = 0.60 

BRAILLE_TO_ENG = {
    '100000': 'a', '110000': 'b', '100100': 'c', '100110': 'd', '100010': 'e',
    '110100': 'f', '110110': 'g', '110010': 'h', '010100': 'i', '010110': 'j',
    '101000': 'k', '111000': 'l', '101100': 'm', '101110': 'n', '101010': 'o',
    '111100': 'p', '111110': 'q', '111010': 'r', '011100': 's', '011110': 't',
    '101001': 'u', '111001': 'v', '010111': 'w', '101101': 'x', '101111': 'y',
    '101011': 'z', '001111': '#', '001001': '-', '010011': '.', '010000': ',',
    '011010': '!', '011001': '?', '001000': "'", '000000': ' ',
    '000001': '', '000101': '', '010101': 'ow'
}

# ---------------------------------------------------------
# 2. OPENCV COMPUTER VISION LOGIC
# ---------------------------------------------------------
def enhance_image_for_yolo(frame):
    """Deepens shadows while keeping whites bright using Gamma Correction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gamma = 2.5 
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    deep_blacks = cv2.LUT(gray, table)
    return cv2.cvtColor(deep_blacks, cv2.COLOR_GRAY2BGR)

def find_page_contour(frame):
    """Finds the 4 corners of the paper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx
    return None

def warp_to_page(frame, contour, padding=50):
    """Flattens the paper and adds a pure black border around the edges."""
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    max_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    new_width = max_width + (padding * 2)
    new_height = max_height + (padding * 2)

    dst = np.array([
        [padding, padding], 
        [new_width - padding - 1, padding], 
        [new_width - padding - 1, new_height - padding - 1], 
        [padding, new_height - padding - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (new_width, new_height))

def process_yolo_boxes(boxes, model_names):
    """Translates the YOLO bounding boxes into English text."""
    if len(boxes) == 0: return "No Braille detected."
    data = boxes.data.cpu().numpy()
    data = data[data[:, 1].argsort()] 
    
    lines, current_line = [], [data[0]]
    for box in data[1:]:
        if abs(box[1] - current_line[-1][1]) < (box[3] * 0.5):
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)

    final_text = ""
    for line in lines:
        line = sorted(line, key=lambda b: b[0]) 
        line_str, prev_x = "", None
        for box in line:
            if prev_x and (box[0] - prev_x) > (box[2] * 1.1):
                line_str += " "
            binary_str = model_names[int(box[5])]
            line_str += BRAILLE_TO_ENG.get(binary_str, "?")
            prev_x = box[0]
        final_text += line_str + "\n"
    return final_text.strip()

# ---------------------------------------------------------
# 3. LOCAL LLM (OLLAMA) AUTOCORRECT
# ---------------------------------------------------------
def autocorrect_with_llm(raw_text):
    """Sends the raw YOLO output to local Llama 3 using Few-Shot Prompting."""
    url = "http://localhost:11434/api/generate"
    
    system_prompt = (
        "You are an expert OCR autocorrect AI. The user will provide raw text scanned from a Braille document. "
        "Because it is an AI vision scan, it contains 'stuttering' duplicate letters (e.g., 'wowowld' instead of 'world'), "
        "missing vowels (e.g., 'hllo' instead of 'hello'), and missing spaces. "
        "Fix the words to form proper English. Output ONLY the corrected text. Do not add quotes or explanations.\n\n"
        "Example 1: hllowowowld -> hello world\n"
        "Example 2: th qick brwn fx -> the quick brown fox\n"
        "Example 3: cmpter scnce -> computer science\n\n"
        f"Raw text: {raw_text}"
    )
    
    payload = {
        "model": "llama3",
        "prompt": system_prompt,
        "stream": False 
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"[WARNING] Ollama failed. Is 'ollama run llama3' running? Error: {e}")
        return raw_text 

# ---------------------------------------------------------
# 4. WEB ROUTES
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/align', methods=['POST'])
def align():
    """Live relay: Draws OpenCV contours on the frame and sends it back."""
    try:
        data = request.json['image'].split(',')[1]
        img_array = np.frombuffer(base64.b64decode(data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        contour = find_page_contour(frame)
        detected = False
        message = "Searching for page edges..."
        
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 4)
            detected = True
            message = "PAGE DETECTED - Ready to capture"
        else:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 4)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
        return jsonify({
            'detected': detected, 
            'message': message,
            'processed_image': processed_base64
        })
            
    except Exception as e:
        return jsonify({'detected': False, 'message': 'Alignment error', 'processed_image': None})

@app.route('/scan', methods=['POST'])
def scan():
    """Heavy lifting: Aligns, Preprocesses, Runs YOLO, and Translates."""
    try:
        data = request.json['image'].split(',')[1]
        img_array = np.frombuffer(base64.b64decode(data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 1. OpenCV Alignment
        contour = find_page_contour(frame)
        final_image = warp_to_page(frame, contour) if contour is not None else frame

        # 2. Contrast Pre-processing (Gamma correction)
        cleaned_image = enhance_image_for_yolo(final_image)

        # 3. YOLO Inference (iou=0.15 aggressively kills duplicate bounding boxes)
        results = model.predict(cleaned_image, conf=CONFIDENCE, iou=0.15, imgsz=640, verbose=False)
        raw_translation = process_yolo_boxes(results[0].boxes, model.names)

        # 4. Send to Ollama for Autocorrect
        print(f"[INFO] YOLO translated: {raw_translation}")
        print("[INFO] Sending to Llama 3 for autocorrect...")
        final_ai_translation = autocorrect_with_llm(raw_translation)

        # Save cropped image for the frontend
        output_path = os.path.join('static', 'latest_crop.jpg')
        cv2.imwrite(output_path, cleaned_image)

        # Send BOTH versions back to the website
        return jsonify({
            'status': 'success', 
            'raw_text': raw_translation,
            'ai_text': final_ai_translation, 
            'image_url': '/static/latest_crop.jpg'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
