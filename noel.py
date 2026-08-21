from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

app = FastAPI()

# --- ตั้งค่า CORS ให้หน้าเว็บเข้ามาคุยได้ ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

client = None

def get_gemini_client():
    global client
    if client is not None:
        return client
    
    api_key = os.environ.get("GEMINI_API_KEY") 
    
    if not api_key:
        raise RuntimeError("กรุณาตั้งค่า GEMINI_API_KEY ก่อนใช้งาน")
    client = genai.Client(api_key=api_key)
    return client

# --- กำหนดโครงสร้างข้อมูลที่จะรับจากหน้าเว็บ ---
class SettingRequest(BaseModel):
    world_setting: str

class ActionRequest(BaseModel):
    action_type: str
    content: str
    history: list         # รับประวัติแชททั้งหมดที่หน้าเว็บส่งมาให้
    world_setting: str    # รับ Setting โลกและตัวละครมาเพื่อย้ำเตือน AI

# --- ฟังก์ชันแปลงประวัติแชทให้อยู่ในรูปแบบที่ Google เข้าใจ ---
def build_history_contents(history):
    return [
        types.Content(
            role=turn["role"],
            parts=[types.Part.from_text(text=turn["text"])],
        )
        for turn in history
    ]

# --- API 1: เริ่มเกมใหม่ ---
@app.post("/start_game")
def start_game(req: SettingRequest):
    # ย้ำเตือน AI ว่ามันคือ Game Master และต้องยึดกฎของโลกนี้
    sys_inst = (
        "คุณคือ Game Master สำหรับเกม Text RPG "
        f"นี่คือข้อมูลของโลกและตัวละคร:\n{req.world_setting}\n"
        "บรรยายเหตุการณ์อย่างมีชีวิตชีวา และอย่าตัดสินใจแทนผู้เล่น"
    )
    
    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="เริ่มเรื่องราว บรรยายฉากเริ่มต้น และทิ้งท้ายให้ผู้เล่นตัดสินใจ",
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.8,
            ),
        )
    except Exception as exc:
        print(f"เกิดข้อผิดพลาดในการเรียก Gemini (start_game): {exc}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {exc}") from exc

    opening_text = response.text or ""
    
    # สร้างประวัติเทิร์นแรก แล้วส่งกลับไปให้หน้าเว็บเก็บไว้
    history = [{"role": "model", "text": opening_text}]
    return {"story": opening_text, "history": history}

# --- API 2: ส่ง Action เล่นเกม ---
@app.post("/action")
def take_action(req: ActionRequest):
    sys_inst = (
        "คุณคือ Game Master สำหรับเกม Text RPG "
        f"นี่คือข้อมูลของโลกและตัวละคร:\n{req.world_setting}\n"
        "บรรยายเหตุการณ์อย่างมีชีวิตชีวา และอย่าตัดสินใจแทนผู้เล่น"
    )
    
    # แปลงคำสั่งให้อ่านง่ายขึ้น
    prompts = {
        "do": f"> ผู้เล่นพยายามที่จะ: {req.content}",
        "talk": f'> ผู้เล่นพูดว่า: "{req.content}"',
        "story": f"> กำหนดให้เหตุการณ์ดำเนินไปดังนี้: {req.content}",
    }
    if req.action_type not in prompts:
        raise HTTPException(status_code=400, detail="Action Type ไม่ถูกต้อง")

    # เอาประวัติจากหน้าเว็บ มาเติมข้อความใหม่ของผู้เล่นเข้าไป
    new_history = req.history.copy()
    new_history.append({"role": "user", "text": prompts[req.action_type]})
    
    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=build_history_contents(new_history), # โยนประวัติทั้งหมดให้ AI อ่าน
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.85,
            ),
        )
    except Exception as exc:
        print(f"เกิดข้อผิดพลาดในการเรียก Gemini (take_action): {exc}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {exc}") from exc

    ai_response = response.text or ""
    
    # เติมคำตอบของ AI ลงในประวัติ แล้วส่งกลับไปให้หน้าเว็บเซฟลงเครื่อง
    new_history.append({"role": "model", "text": ai_response})
    return {"response": ai_response, "history": new_history}

# --- API 3: เช็คสถานะเซิร์ฟเวอร์ ---
@app.get("/health")
def health_check():
    return {"status": "ok"}
