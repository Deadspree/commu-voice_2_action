using System;
using System.Collections;
using UnityEngine;

// Streams 100 ms mono PCM packets while push-to-talk is held; keeps no recording on disk.
public class MicrophoneCapture : MonoBehaviour
{
    public int sampleRate = 16000;
    public int maxSeconds = 20;
    public string deviceName = "";
    public bool IsRecording { get; private set; }
    public event Action Started;
    public event Action<float[], int> Chunk;
    public event Action Ended;
    public event Action<string> Failed;
    AudioClip clip;
    float deadline;
    int generation, readPosition, emitted;
    string Device => string.IsNullOrEmpty(deviceName) ? null : deviceName;

    public void Begin() { Cancel(); StartCoroutine(BeginRecording(++generation)); }
    IEnumerator BeginRecording(int token)
    {
        yield return Application.RequestUserAuthorization(UserAuthorization.Microphone);
        if (token != generation) yield break;
        if (!Application.HasUserAuthorization(UserAuthorization.Microphone))
        { Failed?.Invoke("Microphone permission denied. Enable Unity in macOS Privacy → Microphone."); yield break; }
        if (Microphone.devices.Length == 0) { Failed?.Invoke("No microphone is available"); yield break; }
        clip = Microphone.Start(Device, false, maxSeconds + 1, sampleRate);
        if (clip == null) { Failed?.Invoke("Cannot start the microphone"); yield break; }
        deadline = Time.unscaledTime + maxSeconds;
        readPosition = emitted = 0;
        IsRecording = true;
        Started?.Invoke();
    }

    void Update()
    {
        if (!IsRecording || clip == null) return;
        if (Time.unscaledTime >= deadline) { Finish(); return; }
        Drain(false);
    }

    void Drain(bool final)
    {
        int position = Mathf.Min(Microphone.GetPosition(Device), clip.frequency * maxSeconds);
        if (position <= readPosition) return;
        int packetSize = Mathf.Max(1, clip.frequency / 10);
        while (IsRecording && position > readPosition && (final || position - readPosition >= packetSize))
        {
            int count = Mathf.Min(packetSize, position - readPosition);
            var interleaved = new float[count * clip.channels];
            clip.GetData(interleaved, readPosition);
            var mono = new float[count];
            for (int i = 0; i < count; i++)
                for (int ch = 0; ch < clip.channels; ch++) mono[i] += interleaved[i * clip.channels + ch] / clip.channels;
            readPosition += count;
            // Most devices honor 16 kHz. Adapt devices that return a different capture rate.
            int length = Mathf.Max(1, Mathf.RoundToInt((float)count * 16000 / clip.frequency));
            var output = new float[length];
            for (int i = 0; i < length; i++)
            {
                float index = (float)i * clip.frequency / 16000;
                int low = Mathf.Min((int)index, count - 1), high = Mathf.Min(low + 1, count - 1);
                output[i] = Mathf.Lerp(mono[low], mono[high], index - low);
            }
            emitted += length;
            Chunk?.Invoke(output, 16000);
        }
    }

    public void Finish()
    {
        if (!IsRecording || clip == null) { Cancel(); return; }
        Drain(true);
        int count = emitted;
        Cancel();
        if (count < 1600) { Failed?.Invoke("Speak for at least 0.1 seconds"); return; }
        Ended?.Invoke();
    }

    public void Cancel()
    {
        generation++;
        if (IsRecording) Microphone.End(Device);
        IsRecording = false;
        if (clip != null) { Destroy(clip); clip = null; }
    }
    void OnDisable() { Cancel(); }
}
