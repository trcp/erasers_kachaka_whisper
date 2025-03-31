#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import torch  # GPU有無のチェックに利用
import concurrent.futures
import queue

class FasterWhisperTranscriber(Node):
    def __init__(self):
        super().__init__('faster_whisper_transcriber')
        self.publisher_ = self.create_publisher(String, 'transcription', 10)
        self.get_logger().info('Faster Whisper Transcriber Node Initialized')

        # faster-whisper モデルのロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel("tiny", device=device, compute_type="float32")

        # PyAudio 設定
        self.CHUNK = 1024           # 一回に取得するサンプル数
        self.RATE = 16000           # サンプリングレート（16kHz推奨）
        self.CHANNELS = 1           # モノラル
        self.FORMAT = pyaudio.paInt16  # 16bit PCM

        # 音声入力のセットアップ
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   input=True,
                                   frames_per_buffer=self.CHUNK)

        self.buffer = []            # 音声データの蓄積領域
        self.buffer_duration = 0.0  # バッファに蓄積した秒数
        self.target_duration = 2.0  # 処理対象とするセグメントの長さ（秒）

        # 非同期認識処理用のスレッドプールと結果キュー
        self.threadpool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.result_queue = queue.Queue()

        # 定期的なタイマーコールバックで音声処理＆結果チェックを実施
        self.timer = self.create_timer(0.1, self.process_audio)

    def process_audio(self):
        try:
            # 音声データの取得と正規化（-1.0～+1.0の範囲へ）
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer.append(audio_np)
            self.buffer_duration += self.CHUNK / float(self.RATE)

            # 蓄積時間が目標に達したら、非同期で文字起こしを実行
            if self.buffer_duration >= self.target_duration:
                audio_data = np.concatenate(self.buffer, axis=0)
                self.buffer = []
                self.buffer_duration = 0.0
                # バックグラウンドで文字起こしタスクを開始するために、スレッドプールを使用
                self.threadpool_executor.submit(self.do_transcription, audio_data)

            # 非同期タスクから結果があれば、取り出してROSトピックにパブリッシュ
            try:
                result_text = self.result_queue.get_nowait()
                self.get_logger().info(f"認識結果: {result_text}")
                msg = String()
                msg.data = result_text
                self.publisher_.publish(msg)
            except queue.Empty:
                pass

        except Exception as e:
            self.get_logger().error(f"エラーが発生しました: {str(e)}")

    def do_transcription(self, audio_data):
        try:
            # faster-whisper による文字起こし
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,   # 高速化のため beam_size を調整できます
                language="en"
            )
            text = " ".join([segment.text for segment in segments])
            self.result_queue.put(text)
        except Exception as e:
            self.get_logger().error(f"Transcription error: {str(e)}")

    def destroy(self):
        self.timer.cancel()
        self.threadpool_executor.shutdown(wait=False)
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FasterWhisperTranscriber()
    try:
        # ROS2 の標準的なスピンループでノードを動作させる
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('ノードを停止します...')
    finally:
        node.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
