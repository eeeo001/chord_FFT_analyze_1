import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io # Streamlit 파일 처리용 라이브러리 추가

# --- (1) 함수 정의: 주파수를 MIDI 노트로 변환 ---
def freq_to_midi(frequency):
    """
    주파수(Hz)를 MIDI 노트 번호로 변환합니다. (A4=440Hz 기준, MIDI 69)
    """
    if frequency <= 0:
        return -1
    # MIDI note = 69 + 12 * log2(frequency / 440.0)
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(round(midi_note))

# --- (2) Streamlit web page settings ---
st.set_page_config(layout="wide") # 넓은 레이아웃 설정
st.title("🎶 Chord FFT 분석 및 작곡 정량화 연구")
st.markdown("### 🎙️ 음성 신호를 푸리에 변환으로 분석하여 화음(Chord)을 식별합니다.")

# --- (3) 파일 업로드 위젯 ---
uploaded_file = st.file_uploader("분석할 오디오 파일 (WAV 권장)을 업로드하세요.", type=['wav', 'mp3'])

if uploaded_file is not None:
    
    try:
        # 파일을 메모리에 로드 (librosa는 Streamlit의 업로드 핸들을 직접 처리할 수 있음)
        y, sr = librosa.load(uploaded_file, sr=None) 
        
        # --- 파일 정보 표시 ---
        st.success("🎉 파일 로드 성공!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sampling Rate (sr)", f"{sr} Hz")
        with col2:
            st.metric("Duration", f"{len(y)/sr:.2f} seconds")
        
        # --- 4. FFT 수행 및 스펙트럼 계산 ---
        N = len(y)
        yf = fft(y)
        xf = fftfreq(N, 1/sr)
        
        half_n = N // 2
        xf_positive = xf[:half_n] # 양의 주파수
        yf_positive = np.abs(yf[:half_n]) # 진폭(Magnitude)
        
        st.subheader("📊 주파수 스펙트럼 시각화")
        
        # --- 5. 스펙트럼 시각화 ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf_positive, yf_positive)
        ax.set_title('Frequency Spectrum (Raw)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([20, 2000]) # 20Hz ~ 2000Hz (음악적 주파수 대역)
        ax.grid(True)
        st.pyplot(fig) 

        # --- 6. 피크 식별 및 배음 필터링 (연구의 핵심) ---
        
        # 6-1. 초기 피크 식별
        magnitude_threshold = np.max(yf_positive) * 0.05
        frequency_resolution = sr / N
        min_freq_separation_hz = 10 # 10Hz 이상 떨어진 피크만 초기 식별
        distance_bins = int(min_freq_separation_hz / frequency_resolution)
        
        peak_indices, _ = find_peaks(yf_positive, height=magnitude_threshold, distance=distance_bins)
        peak_frequencies = xf_positive[peak_indices]
        peak_magnitudes = yf_positive[peak_indices]

        # 6-2. 배음 필터링 (Harmonic Filtering)
        initial_sorted_peaks = sorted(zip(peak_magnitudes, peak_frequencies), key=lambda x: x[0], reverse=True)
        filtered_fundamentals = []
        tolerance = 0.015 # 1.5% 오차 허용
        
        for mag, freq in initial_sorted_peaks:
            is_harmonic = False
            for fundamental_freq, fundamental_mag in filtered_fundamentals:
                for n in range(2, 6): # 2차~5차 배음 체크
                    expected_harmonic_freq = fundamental_freq * n
                    if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < tolerance:
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            if not is_harmonic:
                # 배음이 아니면 근음(Fundamental)으로 추가
                filtered_fundamentals.append((freq, mag))

        filtered_fundamentals.sort(key=lambda x: x[0])
        fundamental_frequencies = [f for f, m in filtered_fundamentals]
        fundamental_midi_notes = [freq_to_midi(f) for f in fundamental_frequencies if f > 50] # 50Hz 미만 노이즈 제거

        st.subheader("🎵 근음(Fundamental Frequencies) 분석 결과")
        st.markdown(f"**식별된 근음(Hz):** `{np.round(fundamental_frequencies, 2)}`")
        
        # --- 7. 화음(Chord) 식별 ---
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_templates = {
            'Major': [0, 4, 7], 'Minor': [0, 3, 7], 'Dominant 7th': [0, 4, 7, 10], 
            'Major 7th': [0, 4, 7, 11], 'Minor 7th': [0, 3, 7, 10]
        }
        
        best_match_score = -1
        best_root_midi = -1
        best_chord_type = ""
        identified_chord = "No chord identified."
        
        # **✅ 수정된 로직 시작: 근음 후보를 가장 낮은 음으로 제한합니다.**
        if fundamental_midi_notes:
            # 1. 검출된 음 중 가장 낮은 음(lowest MIDI number)을 찾습니다.
            root_candidate_midi = min(fundamental_midi_notes)
            # 2. 코드 매칭 루프를 이 하나의 근음 후보로만 실행합니다.
            root_midi_candidates = [root_candidate_midi]
        else:
            root_midi_candidates = []
        
        # 식별된 노트들을 음악 이론에 따라 코드 매칭
        # for root_midi in unique_fundamental_midi_notes: # <-- 기존 루프 (삭제)
        for root_midi in root_midi_candidates: # ✅ 수정된 루프
            observed_intervals = set((note - root_midi) % 12 for note in fundamental_midi_notes)

            for chord_type, template_intervals in chord_templates.items():
                match_score = sum(1 for interval in template_intervals if interval in observed_intervals)

                # ✅ 수정: 매칭 점수가 더 높거나 (AND) 매칭 점수가 같고 현재 근음이 가장 낮은 근음일 때 우선합니다.
                if (match_score >= 2 and match_score > best_match_score) or \
                   (match_score >= 2 and match_score == best_match_score and root_midi == root_candidate_midi):
                    
                    best_match_score = match_score
                    best_root_midi = root_midi
                    best_chord_type = chord_type

        # 최종 코드 결과
        if best_root_midi != -1 and best_match_score >= 2:
            root_name = note_names[best_root_midi % 12]
            identified_chord = f"**{root_name} {best_chord_type}**"
        
        st.markdown(f"### 🎼 최종 식별 화음: {identified_chord}")
        st.info(f"매칭 점수 (최대 {len(fundamental_midi_notes)}): {best_match_score}")

    except Exception as e:
        # 오디오 파일이 손상되었거나 형식이 잘못되었을 때
        st.error(f"❌ 오디오 파일 분석 중 오류가 발생했습니다: {e}")
        st.info("지원되는 WAV 또는 MP3 파일인지 확인하고 다시 시도해 보세요.")

else:
    # 파일 업로드 대기 중
    st.info("⬆️ 음성 파일을 업로드하고 분석 결과를 확인하세요. (WAV 파일 권장)")
