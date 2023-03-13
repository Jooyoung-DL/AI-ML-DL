# Dataset : UrbanSound8K
# 목적 : SVM 성능 테스트


##### import lib
import queue
import threading
import pyaudio
import numpy as np
import librosa
from sklearn.svm import SVC
import joblib
import wave
import librosa.display
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SVM
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import xgboost
from sklearn.preprocessing import StandardScaler
from scipy.fft import dct
##### Param Define
# 오디오 데이터 전처리를 위한 파라미터
RATE = 44100  # 초당 샘플의 개수 (44100Hz을 기준으로 함. 44100(샘플개수)/s => 0.00022675 초당 1개의 샘플 생성)
CHANNELS = 1  # 입력 또는 출력의 채널 개수 (모노 마이크를 통한 입력으로 1을 지정)
FRAME_BUFFER = 4410  # 버퍼에 저장할 샘플의 개수
N_MFCC = 12  # 추출할 MFCC의 개수
#HOP_LENGTH = BLOCKSIZE // 4  # Hop length for the spectrogram
pyaudio_format = pyaudio.paFloat32 # 스트림 데이터의 포멧 지정
npaudio_format = np.float32

input_audio_feature = []
record_byte = 2

# 모델 및 라벨을 위한 파라미터
MODEL_PATH = "/home/jylee/src/ml/svm" # 모델 경로 지정

CLASSES_LABEL = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling"
           , "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]  # 클래스 라벨링
##### SVM 모델 로드
# SVM 모델 불러오기
SVM = joblib.load(MODEL_PATH)
##### Save Wav File
def save_wav(s_filename, in_audio):
    save_wav_file = wave.open(s_filename, 'wb')    # 경로와 함께 파일명을 전달 받아야한다.
    save_wav_file.setnchannels(CHANNELS)           # wav 파일의 입력 또는 출력의 채널 수 
    save_wav_file.setsampwidth(record_byte)     # wav 파일의 샘플의 포멧을 지정 
    save_wav_file.setframerate(RATE)               # wav 파일의 프레임 크기를 지정
    save_wav_file.writeframes(b''.join(in_audio))  # 프레임에 데이터 입력
    save_wav_file.close()                          # 종료


def extract_frame(in_audio):
    full_frame = deque(maxlen=10)
    
    for i, x in enumerate(in_audio):
        if len(full_frame) ==10:
            str_frame = b''.join(full_frame)
            pre_extract_features(str_frame)
            
# 스펙트로그램을 계산합니다.
def update(in_audio, spec):
    #y = np.frombuffer(in_audio.read(1024), dtype=np.float32)
    y = in_audio
    spec.set_data(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=44100), ref=np.max))
    return spec
 
fig, ax = plt.subplots()
spec = ax.imshow(np.zeros((128, 128)), cmap='viridis', origin='lower', aspect='auto')
    
def draw_chart_mfccs(in_audio):
    spect = update(in_audio, spec)    
    ani = FuncAnimation(fig, spect, interval=1)
    # 애니메이션을 화면에 출력합니다.
    plt.savefig("/home/jylee/src/ml/ani.png")
    plt.show()
    
##### Classification
def pre_extract_features(str_frame):
    scaler = StandardScaler()
    wav_int = np.frombuffer(str_frame, np.int16) 
    wav_float = wav_int.astype(np.float32)
    #print("wav_float frame: ", wav_float)
    wav_float_path = "/home/jylee/src/ml/test_output/float.wav"
    wav_int_path = "/home/jylee/src/ml/test_output/int.wav"
    save_wav(wav_float_path, wav_float)
    save_wav(wav_int_path, wav_int)
        
    try:
        input_signal, input_sample_rate = librosa.load(wav_int_path) #7061-6-0-0.wav file
        audio_int = wav_int # RATE = 44100
        audio_float = wav_float # RATE = 4410
        n_fft = int(input_sample_rate * 0.025)     # n_fft는 Sample Rate와 Frame Length에 의하여 결정한다. Frame Length는 25ms를 기본적으로 사용한다.
        hop_length = int(input_sample_rate * 0.01) # hop_length = Sample Rate x Frame stride Frame stride의 경우 10ms로 기본적으로 사용한다.  
        stft = librosa.stft(input_signal, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        mfccs = librosa.feature.mfcc(y=input_signal, sr=RATE, S=log_spectrogram, n_mfcc=0)
        dct_mfccs = dct(x=mfccs, type=2, axis=0)
        scaler = scaler.fit_transform(dct_mfccs)
        mfccs_pad = np.pad(scaler, pad_width=((0, 0), (0, 0)), mode='constant')
        mfccsscaled = np.mean(scaler.T,axis=0)
        #draw_chart_mfccs(dct_mfccs, RATE, hop_length)
        #plot_audio(wav_int)
    except Exception as e:
        print("Error encountered ")
        return None
    predict(scaler, mfccs_pad, mfccsscaled)
    #draw_chart_mfccs(wav_float)
    return scaler, mfccs_pad, mfccsscaled

def plot_audio(wav_int) :
    plt.cla()
    plt.axis([0, FRAME_BUFFER * 10, -5000, 5000])
    try:
        plt.plot(wav_int[-FRAME_BUFFER * 10:])
    except:
        plt.plot(wav_int)
    plt.pause(0.01)
            
def predict(mfccs, mfccs_pad, mfccs_scaled):
    input_audio_feature = []
    input_audio_feature.append([mfccs, mfccs_pad, mfccs_scaled])

    input_audio_feature = pd.DataFrame(input_audio_feature, columns=['mfccs', 'mfccs_pad','feature'])

    input_audio_feature.to_pickle("input_audio_feature.pkl")

    # read the variables
    input_audio_feature = pd.read_pickle("input_audio_feature.pkl")

    hop_length = int(FRAME_BUFFER * 0.01)

    #draw_chart_mfccs(mfccs, input_signal, input_sample_rate, hop_length)

    x_input = np.array(input_audio_feature.feature.tolist())


    y_input = int(SVM.predict(x_input))
    label = CLASSES_LABEL[y_input]
    print("pred : ", label)
    #print("y_pred : ",y_input)
    #print("y_pred shape: ",y_input.shape)            
     
##### MIC Stream
class MicrophoneStream(object):
    """마이크 입력 클래스"""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # 마이크 입력 버퍼 생성
        self._buff = queue.Queue()
        self.closed = True

    # 클래스 열면 발생함.
    def __enter__(self):
        # pyaudio 인터페이스 생성
        self._audio_interface = pyaudio.PyAudio()
        # 16비트, 모노로 마이크 열기
        # 여기서 _fill_buffer 함수가 바로 callback함수 인데
        # 실제 버퍼가 쌓이면 이곳이 호출된다.
        # 즉, _fill_buffer 마이크 입력을 _fill_buffer 콜백함수로 전달 받음
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )        
        self.closed = False        
        return self

    def __exit__(self, type, value, traceback):
        # 클래스 종료시 발생
        # pyaudio 종료
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
    
    # 마이크 버퍼가 쌓이면(CHUNK = 1600) 이 함수 호출 됨. 
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # 마이크 입력 받으면 큐에 넣고 리턴
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    # 제너레이터 함수 
    def generator(self):
        #클래스 종료될 떄까지 무한 루프 돌림 
        while not self.closed:
            
            # 큐에 데이터를 기다림.
            # block 상태임.
            
            chunk = self._buff.get()

            # 데이터가 없다면 문제 있음
            if chunk is None:
                return

            # data에 마이크 입력 받기
            data = [chunk]

            # 추가로 받을 마이크 데이터가 있는지 체크 
            while True:
                try:
                    # 데이터가 더 있는지 체크
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    # 데이터 추가
                    data.append(chunk)
                except queue.Empty:
                    # 큐에 데이터가 더이상 없다면 break
                    break

            #마이크 데이터를 리턴해줌 
            yield b''.join(data)
### Main
def main():
    with MicrophoneStream(RATE, FRAME_BUFFER) as stream:
        audio_generator = stream.generator()
        extract_frame(audio_generator)
        #pre_extract_features(audio_generator)
        #print(list(audio_generator))



if __name__ == '__main__':
    main()
    print("end main")
