import gradio as gr
from deepface import DeepFace
import cv2
import os
import requests
import numpy as np
import instaloader
import hashlib
import time
from pathlib import Path
from functools import lru_cache

# ------------------- कैश सेटअप -------------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 3600  # 1 घंटा

def get_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def get_cached_image(url: str):
    key = get_cache_key(url)
    path = CACHE_DIR / f"{key}.jpg"
    if path.exists():
        if time.time() - path.stat().st_mtime < CACHE_TTL:
            img = cv2.imread(str(path))
            if img is not None:
                return img
    return None

def set_cached_image(url: str, img: np.ndarray):
    key = get_cache_key(url)
    path = CACHE_DIR / f"{key}.jpg"
    cv2.imwrite(str(path), img)

# ------------------- सोशल मीडिया फेचर -------------------
def fetch_photo(platform: str, username: str):
    username = username.strip()
    
    if platform == "GitHub":
        url = f"https://api.github.com/users/{username}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("avatar_url")
        except:
            pass
            
    elif platform == "Instagram":
        # Instaloader (बिना लॉगिन)
        try:
            L = instaloader.Instaloader()
            profile = instaloader.Profile.from_username(L.context, username)
            return profile.profile_pic_url
        except:
            pass
        # Fallback: Instagram public API
        try:
            url = f"https://www.instagram.com/{username}/?__a=1"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()["graphql"]["user"]["profile_pic_url_hd"]
        except:
            pass
            
    elif platform == "Facebook":
        # Facebook Graph API (बिना टोकन)
        return f"https://graph.facebook.com/{username}/picture?type=large&width=500&height=500"
        
    elif platform == "Twitter":
        # Unavatar.io – मुफ्त और भरोसेमंद
        return f"https://unavatar.io/twitter/{username}"
        
    elif platform == "LinkedIn":
        # Unavatar.io – LinkedIn के लिए
        return f"https://unavatar.io/linkedin/{username}"
        
    elif platform == "YouTube":
        # YouTube channel ID (उदाहरण: UCXuqSBlHAE6Xw-yeJA0Tunw)
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if api_key:
            url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={username}&key={api_key}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    if items:
                        return items[0]["snippet"]["thumbnails"]["high"]["url"]
            except:
                pass
        return None
        
    return None

# ------------------- मुख्य वेरिफिकेशन -------------------
def verify_face(live_image, platform, username, model_name="VGG-Face", detector="mtcnn"):
    if live_image is None:
        return None, "⚠️ कृपया वेबकैम से फोटो खींचें।"
    if not username:
        return None, "⚠️ कृपया यूज़रनेम दर्ज करें।"
    
    # 1. सोशल फोटो का URL पता करें
    img_url = fetch_photo(platform, username)
    if not img_url:
        return None, f"❌ {platform} पर '{username}' नहीं मिला। कृपया यूज़रनेम चेक करें।"
    
    # 2. कैश से या डाउनलोड करके फोटो लें
    social_img = get_cached_image(img_url)
    if social_img is None:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(img_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None, f"❌ फोटो डाउनलोड नहीं हो पाई (स्टेटस {resp.status_code})"
            img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            social_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if social_img is None:
                return None, "❌ डाउनलोड की गई फाइल मान्य फोटो नहीं है।"
            set_cached_image(img_url, social_img)
        except Exception as e:
            return None, f"❌ डाउनलोड में त्रुटि: {str(e)}"
    
    # 3. लाइव इमेज को BGR में बदलें
    live_bgr = cv2.cvtColor(live_image, cv2.COLOR_RGB2BGR)
    
    # 4. DeepFace से तुलना करें
    try:
        result = DeepFace.verify(
            img1_path=live_bgr,
            img2_path=social_img,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=True
        )
        
        distance = result["distance"]
        verified = result["verified"]
        threshold = 0.40
        match_score = max(0, min(100, (1 - (distance / threshold)) * 100))
        
        social_rgb = cv2.cvtColor(social_img, cv2.COLOR_BGR2RGB)
        
        # 5. परिणाम संदेश
        status = "✅ सफल" if verified else "❌ असफल"
        emoji = "🎉" if verified else "😞"
        msg = f"""### {emoji} वेरिफिकेशन {status}

| मीट्रिक | मान |
| :--- | :--- |
| **प्लेटफॉर्म** | {platform} |
| **यूज़रनेम** | {username} |
| **मॉडल** | {model_name} |
| **डिटेक्टर** | {detector} |
| **मैच स्कोर** | **{match_score:.2f}%** |
| **दूरी** | {distance:.4f} |
| **थ्रेशोल्ड** | {threshold} |
"""
        return social_rgb, msg
        
    except ValueError as ve:
        return None, f"⚠️ चेहरा पहचान में त्रुटि: {ve}\n\n**सुझाव:** सीधे कैमरे में देखें, रोशनी अच्छी हो, और चेहरा स्पष्ट दिखे।"
    except Exception as e:
        return None, f"❌ तकनीकी त्रुटि: {str(e)}"

# ------------------- Gradio UI -------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Face Verification") as demo:
    gr.Markdown("""# 🌐 यूनिवर्सल फेस वेरिफिकेशन
    अपने लाइव फोटो को किसी भी सोशल मीडिया प्रोफाइल फोटो से मिलाएं (बिना लॉगिन के!)""")
    
    with gr.Row():
        with gr.Column(scale=1):
            live_input = gr.Image(
                sources=["webcam", "upload"], 
                type="numpy", 
                label="📸 लाइव फोटो या अपलोड",
                height=300
            )
            
            platform = gr.Dropdown(
                choices=["GitHub", "Instagram", "Facebook", "Twitter", "LinkedIn", "YouTube"],
                value="GitHub",
                label="🌍 प्लेटफॉर्म चुनें"
            )
            
            username = gr.Textbox(
                label="🆔 यूज़रनेम / Channel ID",
                placeholder="e.g., torvalds, zuck, pmmodi",
                value="torvalds"
            )
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=["VGG-Face", "Facenet", "ArcFace", "Dlib", "SFace"],
                    value="VGG-Face",
                    label="🧠 मॉडल"
                )
                detector_choice = gr.Dropdown(
                    choices=["mtcnn", "opencv", "retinaface"],
                    value="mtcnn",
                    label="🎯 डिटेक्टर"
                )
            
            verify_btn = gr.Button("🚀 वेरिफाई करें", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            social_img = gr.Image(
                label="🖼️ सोशल मीडिया प्रोफाइल फोटो",
                height=300
            )
            result_text = gr.Markdown(label="📝 परिणाम")
    
    verify_btn.click(
        fn=verify_face,
        inputs=[live_input, platform, username, model_choice, detector_choice],
        outputs=[social_img, result_text]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ कैसे काम करता है?
    1. अपना **लाइव फोटो** वेबकैम से खींचें या अपलोड करें।
    2. **प्लेटफॉर्म** और **यूज़रनेम** डालें।
    3. "वेरिफाई करें" पर क्लिक करें – ऐप स्वतः उस प्रोफाइल की पब्लिक फोटो खोजकर मिलान करेगा।
    
    **नोट:** Instagram और LinkedIn के लिए कभी-कभी फोटो न मिले तो GitHub/Facebook/Twitter ट्राई करें (ये 100% काम करते हैं)।
    """)

# ------------------- Render के लिए लॉन्च -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
