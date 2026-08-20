from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

app = FastAPI()

# --- ก๊อปปี้บล็อกนี้ไปวาง ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้ทุกหน้าเว็บเข้ามาคุยได้
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตให้รับคำสั่ง OPTIONS, POST, GET ได้ทั้งหมด
    allow_headers=["*"],
)
# ------------------------

client = None
# ... (โค้ดเดิมของคุณต่อจากนี้) ...
client = None


def get_gemini_client():
    global client
    if client is not None:
        return client
    
    # ❌ ลบบรรทัดนี้ทิ้ง: api_key = os.environ.get("GEMINI_API_KEY")
    
    # ✅ เปลี่ยนมาใส่แบบนี้แทน (เอาคีย์จริงๆ ของคุณมาใส่ในเครื่องหมายคำพูด)
    api_key = "GEMINI_API_KEY" 
    
    if not api_key:
        raise RuntimeError("กรุณาตั้งค่า GEMINI_API_KEY ก่อนใช้งาน")
    client = genai.Client(api_key=api_key)
    return client


game_state = {
    "world_setting": "",
    "system_instruction": "",
    "history": [],
}


class SettingRequest(BaseModel):
    world_setting: str


class ActionRequest(BaseModel):
    action_type: str
    content: str


class EditRequest(BaseModel):
    index: int
    new_text: str


def build_history_contents(history):
    return [
        types.Content(
            role=turn["role"],
            parts=[types.Part.from_text(text=turn["text"])],
        )
        for turn in history
    ]


@app.post("/start_game")
def start_game(req: SettingRequest):
    game_state["history"] = []
    game_state["world_setting"] = req.world_setting
    game_state["system_instruction"] = (
        "คุณคือ Game Master สำหรับเกม Text RPG "
        f"นี่คือ Setting ของโลก: {req.world_setting}\n"
        "บรรยายเหตุการณ์อย่างมีชีวิตชีวา และอย่าตัดสินใจแทนผู้เล่น"
    )
    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="เริ่มเรื่องราว บรรยายฉากเริ่มต้น และทิ้งท้ายให้ผู้เล่นตัดสินใจ",
            config=types.GenerateContentConfig(
                system_instruction=game_state["system_instruction"],
                temperature=0.8,
            ),
        )
    # แก้ไขในส่วนของ @app.post("/start_game")
    except Exception as exc:
        print(f"เกิดข้อผิดพลาดในการเรียก Gemini (start_game): {exc}") # พิมพ์ลง Terminal
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(exc)}")

    opening_text = response.text or ""
    game_state["history"].append({"role": "model", "text": opening_text})
    return {"story": opening_text, "history": game_state["history"]}


@app.post("/action")
def take_action(req: ActionRequest):
    if not game_state["history"]:
        raise HTTPException(status_code=400, detail="กรุณาเริ่มเกมก่อน")

    prompts = {
        "do": f"> ผู้เล่นพยายามที่จะ: {req.content}",
        "talk": f'> ผู้เล่นพูดว่า: "{req.content}"',
        "story": f"> กำหนดให้เหตุการณ์ดำเนินไปดังนี้: {req.content}",
    }
    if req.action_type not in prompts:
        raise HTTPException(status_code=400, detail="Action Type ไม่ถูกต้อง")

    game_state["history"].append({"role": "user", "text": prompts[req.action_type]})
    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=build_history_contents(game_state["history"]),
            config=types.GenerateContentConfig(
                system_instruction=game_state["system_instruction"],
                temperature=0.85,
            ),
        )
    # แก้ไขในส่วนของ @app.post("/action")
    except Exception as exc:
        print(f"เกิดข้อผิดพลาดในการเรียก Gemini (take_action): {exc}") # พิมพ์ลง Terminal
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(exc)}")

    ai_response = response.text or ""
    game_state["history"].append({"role": "model", "text": ai_response})
    return {"response": ai_response, "history": game_state["history"]}


@app.post("/edit_story")
def edit_story(req: EditRequest):
    if req.index < 0 or req.index >= len(game_state["history"]):
        raise HTTPException(status_code=400, detail="Index ไม่ถูกต้อง")
    game_state["history"][req.index]["text"] = req.new_text
    return {"message": "แก้ไขเรื่องราวสำเร็จ", "history": game_state["history"]}


@app.get("/health")
def health_check():
    return {"status": "ok"}
