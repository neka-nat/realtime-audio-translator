import asyncio
import base64
import json
import os
import numpy as np
import soundcard as sc
from dotenv import load_dotenv
import websockets

load_dotenv()

API_KEY = os.environ.get('OPENAI_API_KEY')

# WebSocket URLとヘッダー情報
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = {
    "Authorization": "Bearer "+ API_KEY, 
    "OpenAI-Beta": "realtime=v1"
}

# 音声設定
CHUNK = 2048  # 音声データのチャンクサイズ
RATE = 24000  # サンプリングレート（24kHz）
CHANNELS = 1  # モノラル

async def send_audio(websocket, speaker):
    """PCの音声を取得して送信する関数"""
    def read_audio_block():
        """同期的に音声データを読み取る関数"""
        try:
            # PCの音声を取得
            data = speaker.record(numframes=CHUNK)
            # モノラルに変換
            data = data[:, 0]
            # float32からint16に変換
            audio_data = (data * 32767).astype(np.int16).tobytes()
            return audio_data
        except Exception as e:
            print(f"音声読み取りエラー: {e}")
            return None

    print("PCの音声を取得して送信中...")
    while True:
        # PCから音声を取得
        audio_data = await asyncio.get_event_loop().run_in_executor(None, read_audio_block)
        if audio_data is None:
            continue
        
        # PCM16データをBase64にエンコード
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        audio_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }

        # WebSocketで音声データを送信
        await websocket.send(json.dumps(audio_event))
        await asyncio.sleep(0)


async def receive_transcript(websocket):
    """翻訳テキストを受信して表示する関数"""
    while True:
        # サーバーからの応答を受信
        response = await websocket.recv()
        response_data = json.loads(response)

        # 翻訳テキストをリアルタイムで表示
        if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
            print(response_data["delta"], end="", flush=True)
        elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
            print("\n---", flush=True)


async def stream_audio_and_translate():
    # デフォルトスピーカーの取得（ループバックを含む）
    speaker = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    
    # WebSocketに接続
    async with websockets.connect(WS_URL, additional_headers=HEADERS) as websocket:
        print("WebSocketに接続しました。")

        # 初期リクエスト（翻訳設定）
        init_request = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],  # テキストのみを要求
                "instructions": "これからの会話で、流れてくる英語の音声をリアルタイムに日本語に翻訳して返してください。",
            }
        }
        await websocket.send(json.dumps(init_request))
        print("初期リクエストを送信しました。")
        
        try:
            with speaker.recorder(samplerate=RATE) as mic:
                # 音声送信タスクと翻訳受信タスクを非同期で並行実行
                send_task = asyncio.create_task(send_audio(websocket, mic))
                receive_task = asyncio.create_task(receive_transcript(websocket))

                # タスクが終了するまで待機
                await asyncio.gather(send_task, receive_task)

        except KeyboardInterrupt:
            print("\n終了します...")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(stream_audio_and_translate())
