#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyaudio
import numpy as np
import time

class WhisperTranscriber(Node):
    def __init__(self):
        super().__init__('whisper_transcriber')
        self.publisher_ = self.create_publisher(String, 'transcription', 10)
        self.get_logger().info('Whisper Transcriber Node Initialized')

        # Whisperモデルのロード
        self.model = whisper.load_model("large")
        
        # PyAudio設定
        self.CHUNK = 1024        # 1回に取得する音声データのサイズ
        self.RATE = 16000        # サンプリングレート（Whisper推奨: 16kHz）
        self.CHANNELS = 1        # モノラル
        self.FORMAT = pyaudio.paInt16  # 16bit PCM

        # 音声入力のセットアップ
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE,
                                   input=True, frames_per_buffer=self.CHUNK)

        self.buffer = []          # 音声データを蓄積するバッファ
        self.buffer_duration = 0.0  # 蓄積した時間（秒）
        self.target_duration = 5.0  # 分析対象とする音声の長さ（秒）

        # タイマーコールバックで定期的に処理を行う
        self.timer = self.create_timer(0.1, self.process_audio)

    def process_audio(self):
        try:
            # 音声データを取得（bytes形式）
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            # numpy配列に変換（int16 -> float32）し、正規化
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer.append(audio_np)
            self.buffer_duration += self.CHUNK / float(self.RATE)

            # 指定した秒数分のデータが蓄積されたら処理
            if self.buffer_duration >= self.target_duration:
                # バッファ内の音声を結合
                audio_data = np.concatenate(self.buffer, axis=0)
                # Whisperが処理しやすいようにデータを調整
                audio_data = whisper.pad_or_trim(audio_data)
                
                # Whisperで文字起こし（例: 英語優先）
                result = self.model.transcribe(audio_data, language="en")
                text = result.get("text", "")

                self.get_logger().info(f"認識結果: {text}")
                # トピックに文字起こし結果をパブリッシュ
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)

                # バッファをクリア
                self.buffer = []
                self.buffer_duration = 0.0

        except Exception as e:
            self.get_logger().error(f"エラーが発生しました: {str(e)}")

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperTranscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('ノードを停止します...')
    finally:
        node.destroy()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
