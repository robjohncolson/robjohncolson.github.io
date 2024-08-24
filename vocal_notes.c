#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <mmsystem.h>
#include <dsound.h>
#include <conio.h>

#pragma comment(lib, "dsound.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "winmm.lib")

#define SAMPLE_RATE 44100
#define BUFFER_SIZE (SAMPLE_RATE * 5)  // 5 seconds of audio
#define M_PI 3.14159265358979323846
#define NUM_NOTES 37  // C3 to C6

LPDIRECTSOUND pDS = NULL;
LPDIRECTSOUNDBUFFER pDSBuffer = NULL;
DWORD bufferOffset = 0;

// Detuned note frequencies (Hz) for C3 to C6
double notes[NUM_NOTES] = {
    128.279, 135.936, 144.030, 152.582, 161.618, 171.166, 181.255, 192.015, 203.479, 215.680, 228.654, 242.440, // C3 to B3
    256.564, 271.873, 288.060, 305.164, 323.236, 342.332, 362.510, 384.030, 406.959, 431.359, 457.308, 484.881, // C4 to B4
    513.139, 543.746, 576.119, 610.327, 646.472, 684.663, 725.019, 768.059, 813.918, 862.718, 914.616, 969.761, // C5 to B5
    1026.29 // C6
};

const char* note_names[NUM_NOTES] = {
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6"
};

int current_note_index = 21; // A4 (432 Hz)
int tempo = 60; // beats per minute
double note_duration = 1.0; // in beats
double silence_duration = 0.5; // in beats
BOOL is_playing = FALSE;

BOOL initialize_directsound(HWND hWnd) {
    HRESULT hr;

    hr = DirectSoundCreate(NULL, &pDS, NULL);
    if (FAILED(hr)) {
        printf("Failed to create DirectSound object. Error code: 0x%lx\n", hr);
        return FALSE;
    }

    hr = IDirectSound_SetCooperativeLevel(pDS, hWnd, DSSCL_PRIORITY);
    if (FAILED(hr)) {
        printf("Failed to set cooperative level. Error code: 0x%lx\n", hr);
        return FALSE;
    }

    WAVEFORMATEX wfx = { 0 };
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = 1;
    wfx.nSamplesPerSec = SAMPLE_RATE;
    wfx.wBitsPerSample = 16;
    wfx.nBlockAlign = (wfx.wBitsPerSample / 8) * wfx.nChannels;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

    DSBUFFERDESC dsbd = { 0 };
    dsbd.dwSize = sizeof(DSBUFFERDESC);
    dsbd.dwFlags = DSBCAPS_GLOBALFOCUS | DSBCAPS_CTRLVOLUME;
    dsbd.dwBufferBytes = BUFFER_SIZE * 2;  // 16-bit samples
    dsbd.lpwfxFormat = &wfx;

    hr = IDirectSound_CreateSoundBuffer(pDS, &dsbd, &pDSBuffer, NULL);
    if (FAILED(hr)) {
        printf("Failed to create sound buffer. Error code: 0x%lx\n", hr);
        return FALSE;
    }

    printf("DirectSound initialized successfully.\n");
    return TRUE;
}

void cleanup_directsound() {
    if (pDSBuffer) {
        IDirectSoundBuffer_Stop(pDSBuffer);
        IDirectSoundBuffer_Release(pDSBuffer);
    }
    if (pDS) IDirectSound_Release(pDS);
    printf("DirectSound cleaned up.\n");
}

BOOL fill_buffer(double frequency, double duration, BOOL is_silence) {
    LPVOID pAudioPtr1 = NULL, pAudioPtr2 = NULL;
    DWORD AudioBytes1 = 0, AudioBytes2 = 0;
    HRESULT hr;

    DWORD totalBytes = (DWORD)(SAMPLE_RATE * duration * 2); // 16-bit samples
    if (totalBytes > BUFFER_SIZE * 2) {
        printf("Warning: Requested duration exceeds buffer size. Truncating.\n");
        totalBytes = BUFFER_SIZE * 2;
    }

    hr = IDirectSoundBuffer_Lock(pDSBuffer, bufferOffset, totalBytes, 
                                 &pAudioPtr1, &AudioBytes1, &pAudioPtr2, &AudioBytes2, 0);
    if (FAILED(hr)) {
        printf("Failed to lock sound buffer. Error code: 0x%lx\n", hr);
        return FALSE;
    }

    short* pSample1 = (short*)pAudioPtr1;
    short* pSample2 = (short*)pAudioPtr2;
    DWORD SampleCount1 = AudioBytes1 / 2;
    DWORD SampleCount2 = AudioBytes2 / 2;

    for (DWORD i = 0; i < SampleCount1; i++) {
        double time = (double)(bufferOffset / 2 + i) / SAMPLE_RATE;
        pSample1[i] = is_silence ? 0 : (short)(32767 * sin(2 * M_PI * frequency * time));
    }

    for (DWORD i = 0; i < SampleCount2; i++) {
        double time = (double)(bufferOffset / 2 + SampleCount1 + i) / SAMPLE_RATE;
        pSample2[i] = is_silence ? 0 : (short)(32767 * sin(2 * M_PI * frequency * time));
    }

    hr = IDirectSoundBuffer_Unlock(pDSBuffer, pAudioPtr1, AudioBytes1, pAudioPtr2, AudioBytes2);
    if (FAILED(hr)) {
        printf("Failed to unlock sound buffer. Error code: 0x%lx\n", hr);
        return FALSE;
    }

    bufferOffset = (bufferOffset + AudioBytes1 + AudioBytes2) % (BUFFER_SIZE * 2);
    return TRUE;
}

BOOL play_continuous() {
    if (!is_playing) {
        HRESULT hr = IDirectSoundBuffer_Play(pDSBuffer, 0, 0, DSBPLAY_LOOPING);
        if (FAILED(hr)) {
            printf("Failed to start sound playback. Error code: 0x%lx\n", hr);
            return FALSE;
        }
        is_playing = TRUE;
        printf("Playback started.\n");
    }
    return TRUE;
}

void stop_playing() {
    if (is_playing) {
        HRESULT hr = IDirectSoundBuffer_Stop(pDSBuffer);
        if (FAILED(hr)) {
            printf("Failed to stop sound playback. Error code: 0x%lx\n", hr);
        } else {
            is_playing = FALSE;
            printf("Playback stopped.\n");
        }
    }
}

BOOL play_note_and_silence(double frequency) {
    double note_seconds = (60.0 / tempo) * note_duration;
    double silence_seconds = (60.0 / tempo) * silence_duration;
    
    if (!fill_buffer(frequency, note_seconds, FALSE)) return FALSE;
    if (!fill_buffer(0, silence_seconds, TRUE)) return FALSE;
    
    return play_continuous();
}

void print_menu() {
    printf("\nCurrent note: %s (%.2f Hz)", note_names[current_note_index], notes[current_note_index]);
    printf("\nTempo: %d BPM", tempo);
    printf("\nNote duration: %.2f beats", note_duration);
    printf("\nSilence duration: %.2f beats", silence_duration);
    printf("\nTotal sequence duration: %.2f beats", note_duration + silence_duration);
    printf("\nPlayback status: %s", is_playing ? "Playing" : "Stopped");
    printf("\nUp/Down: Change note");
    printf("\n+/-: Adjust tempo");
    printf("\n[/]: Adjust note duration");
    printf("\n,/.: Adjust silence duration");
    printf("\nSpace: Play current note sequence");
    printf("\ns: Start/Stop audio output");
    printf("\nq: Quit");
    printf("\nEnter choice: ");
}

int main() {
    HWND hWnd = GetConsoleWindow();
    if (!initialize_directsound(hWnd)) {
        printf("Failed to initialize DirectSound. Press any key to exit.\n");
        _getch();
        return 1;
    }

    int choice;
    do {
        print_menu();
        choice = _getch();
        
        if (choice == 0 || choice == 224) { // Arrow key pressed
            choice = _getch(); // Get the actual arrow key code
            switch(choice) {
                case 72: // Up arrow
                    current_note_index = (current_note_index < NUM_NOTES - 1) ? current_note_index + 1 : current_note_index;
                    if (is_playing && !play_note_and_silence(notes[current_note_index])) {
                        printf("Failed to play note. Press any key to continue.\n");
                        _getch();
                    }
                    break;
                case 80: // Down arrow
                    current_note_index = (current_note_index > 0) ? current_note_index - 1 : current_note_index;
                    if (is_playing && !play_note_and_silence(notes[current_note_index])) {
                        printf("Failed to play note. Press any key to continue.\n");
                        _getch();
                    }
                    break;
            }
        } else {
            switch(choice) {
                case '+':
                    tempo = (tempo < 240) ? tempo + 5 : tempo;
                    break;
                case '-':
                    tempo = (tempo > 42) ? tempo - 5 : tempo;
                    break;
                case '[':
                    note_duration = (note_duration > 0.0625) ? note_duration - 0.0625 : note_duration;
                    if (note_duration + silence_duration > 12) silence_duration = 12 - note_duration;
                    break;
                case ']':
                    note_duration = (note_duration < 12 && note_duration + silence_duration < 12) ? note_duration + 0.0625 : note_duration;
                    break;
                case ',':
                    silence_duration = (silence_duration > 0.0625) ? silence_duration - 0.0625 : silence_duration;
                    break;
                case '.':
                    silence_duration = (silence_duration < 12 && note_duration + silence_duration < 12) ? silence_duration + 0.0625 : silence_duration;
                    break;
                case ' ':
                    if (!play_note_and_silence(notes[current_note_index])) {
                        printf("Failed to play note. Press any key to continue.\n");
                        _getch();
                    }
                    break;
                case 's':
                    if (is_playing) {
                        stop_playing();
                    } else {
                        if (!play_note_and_silence(notes[current_note_index])) {
                            printf("Failed to start playback. Press any key to continue.\n");
                            _getch();
                        }
                    }
                    break;
            }
        }
    } while (choice != 'q');

    stop_playing();
    cleanup_directsound();
    printf("Program ended. Press any key to exit.\n");
    _getch();
    return 0;
}