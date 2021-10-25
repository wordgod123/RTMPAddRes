from genericpath import exists
import queue
import threading
import cv2
import subprocess as sp
import time
from numpy.lib import add_docstring
import pyaudio
import wave
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import ffmpeg


class Live(object):
    def __init__(self, inputUrl="", rtmpUrl=""):
        self.frame_queue = queue.Queue()
        self.video_queue = queue.Queue() # 待播放的视频列表
        self.playing_video = None # 正在播放的视频
        self.command = ""
        # 自行设置
        self.rtmpUrl = rtmpUrl
        self.camera_path = inputUrl
        self.width = None
        self.hight = None

        #音频
        self.audio = Audio("test.wav")
        self.audio.add_radio("test1.wav")

    def read_frame(self):
        print("开启推流")
        if self.camera_path:
            cap = cv2.VideoCapture(self.camera_path)
        else:
            cap = cv2.VideoCapture(0)   

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.hight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        # ffmpeg command
        self.command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-ar', '44100',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
                '-listen', '1',
                # self.rtmpUrl
                # "cc.mp4"
                "rtmp://127.0.0.1:3000/live/cc"
                ]

        # read webcamera
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            # put frame into queue
            self.frame_queue.put(frame)


            # 获取视频
            # cv2.imshow('capture', frame)
            # if cv2.waitKey(1) & 0XFF == ord("q"):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     break
            # else:
            #     time.sleep(1)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break

        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                # process frame
                # 你处理图片的代码
                # write to pipe
                # print(frame)

                # 添加图片
                frame = self.merge_image(frame, scale_rate=0.3, position=[500,100])


                # 合并视频贞
                add_frame = self.get_video_frame()
                # print(type(add_frame))
                if type(add_frame) == np.ndarray:
                    frame = self.merge_frame(frame, add_frame)

                radio = self.audio.get_add_frame(self.audio.CHUNK)
                if radio:
                    p.stdin.write(radio)

                p.stdin.write(frame.tostring())

    def merge_frame(self, frame, addFrame, scale_rate=0.3, position=[0, 0]):
        img1 = Image.fromarray(frame)
        img2 = Image.fromarray(addFrame)
        img2 = img2.resize((int(self.width * scale_rate), int(self.hight * scale_rate)))
        img1.paste(img2, position)
        return np.asarray(img1)

    def merge_image(self, frame, scale_rate=0.3, position=None):
        o = Image.open("test.jpg")
        o = o.resize((int(self.width * scale_rate), int(self.hight * scale_rate)))
        img1 = Image.fromarray(frame)
        r,g,b = img1.split()
        # img1 = Image.merge("RGB", (r, g, b))
        if isinstance(position, list) and len(position) == 2:
            img1.paste(o, position)

        out = np.asarray( img1)
        return out
        
    def add_video(self, video_path=None):
        # 打开视频，并存入播放视频的队列中
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            self.video_queue.put(cap)
    def get_video_frame(self):
        if self.playing_video and self.playing_video.isOpened():
            ret, frame = self.playing_video.read()
            if ret == True:
                return frame
        # 视频已经播放完了，则释放资源，并将正播放视频值为空
        if self.playing_video:
            self.playing_video.release()
            self.playing_video = None
        # 检查待播放列表是否为空，如果不为空，则添加到正播放中
        if not self.video_queue.empty():
            self.playing_video = self.video_queue.get()
            return self.get_video_frame()
        return None

    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]

        #  添加视频
        # self.add_video("test.mp4")
        for i in threads:
            i.join()

class Audio(object):
    def __init__(self,wave_out_path):
        self.waiting_play_queue = queue.Queue() #待播放音频列表
        self.playing = None # 正播放的音频

        self.p = pyaudio.PyAudio()

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1 # 麦克风的channel
        self.RATE = 44100

        self.wf = wave.open(wave_out_path, 'wb')
        self.wf.setnchannels(self.CHANNELS) # 1 单声道, 2 立体声
        self.wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        self.wf.setframerate(self.RATE)

    def __del__(self):
        self.wf.close()

    def add_radio(self, radio_path):
        #  添加音频源
        if radio_path and os.path.exists(radio_path):
            wf = wave.open(radio_path, 'rb')
            self.waiting_play_queue.put(wf)

    def get_add_frame(self, frame_count):
        if self.playing:
            data2 = self.playing.readframes(frame_count)
            # 如果当前的播放结束，则情况播放列表
            if not data2:
                self.playing = None
            else:
                return data2
        # 检查是否有待播放的队列
        if not self.waiting_play_queue.empty():
            self.playing = self.waiting_play_queue.get()
            return self.get_add_frame(frame_count)

    def record_audio(self,record_second):
        stream = None
        def callback(in_data, frame_count, time_info, status):
            # 获取额外播放音频信息
            add_frame = self.get_add_frame(frame_count)
            if add_frame:
                decoded_add = np.frombuffer(add_frame, np.int16)
                decoded_in = np.frombuffer(in_data, np.int16)
                newdata = (decoded_in * 0.7 + decoded_add* 0.3).astype(np.int16)
                self.wf.writeframes(newdata)
                return (newdata.tostring(), status)
            else:
                self.wf.writeframes(in_data)
                return (in_data, status)

        stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=callback)

        # out_stream = p.open(format=FORMAT,   # 播放扬声器
        #                 channels=CHANNELS,
        #                 rate=RATE,
        #                 output=True,
        #                 frames_per_buffer=CHUNK)
        print("* recording")
        stream.start_stream()
        print("* done recording")
        while stream.is_active():
            time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        self.p.terminate()
        self.wf.close()

    def mac(self):
        print(pyaudio.PaMacCoreStreamInfo().get_channel_map())

    def get_audio_devices_info(self):
        # 获取到扬声器的channel
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print((i,dev['name'],dev['maxInputChannels']))

class ReadRTMP(object):
    def ReadFromRTMP(self):
        self.source = ffmpeg.input("rtmp://127.0.0.1:3000/live/cc")
        audio = self.source.audio.filter("aecho", 0.8, 0.9, 1000, 0.3)
        video = self.source.video.hflip()
        out = ffmpeg.output(audio, video, "xx.mp4")
        ffmpeg.run(out)
    def get_viedo_width_height(self,path):
        cap = cv2.VideoCapture(path)

        # Get video information
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    def split_av(self):
        width, height = self.get_viedo_width_height("test.mp4")
        process1 = (
            ffmpeg
            .input("test.mp4")
            .output('pipe:', format='rawvideo')
            .run_async(pipe_stdout=True)
        )
        process2 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output("zz.mp4", pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        while True:
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
            )
            out_frame = in_frame * 0.3
            # process2.stdin.write(
            #     out_frame
            #     .astype(np.uint8)
            #     .tobytes()
            # )
            process2.stdin.write(in_bytes)
        process2.stdin.close()
        process1.wait()
        process2.wait()


if __name__ == "__main__":
    url = None
    live = Live(inputUrl=url, rtmpUrl="rtmp://localhost/live/cc")
    live.run()
    # audio = Audio("test.wav")
    # audio.add_radio("test1.wav")
    # audio.get_audio_devices_info()
    # audio.record_audio(record_second=4)
    # audio.mac()
    # a = ReadRTMP()
    # a.split_av()