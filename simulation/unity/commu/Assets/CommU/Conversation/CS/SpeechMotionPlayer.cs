using System;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(150)]
public class SpeechMotionPlayer : MonoBehaviour
{
    public CS_TestRotateBones target;
    [Tooltip("Audio scheduling lead time; separate from the response buffer below.")]
    public double prebufferSeconds = .15;
    [Min(0)] public float startupBufferSeconds = 4;
    [Min(0)] public float rebufferSeconds = 4;
    public bool logTiming = true;
    public float returnSpeed = 120;
    public bool IsBusy => responseActive || current != null || next != null || queue.Count > 0;
    public bool IsBuffering => responseActive && (current == null || AudioSettings.dspTime < current.start);
    public bool IsRebuffering => IsBuffering && UnderrunCount > 0;
    public bool OwnsController => ownsTarget;
    public int UnderrunCount { get; private set; }
    public double BufferedSeconds
    {
        get
        {
            double now = AudioSettings.dspTime;
            return queuedSeconds + (current == null ? 0 : Math.Max(0, current.End - Math.Max(now, current.start))) +
                (next == null ? 0 : Math.Max(0, next.End - Math.Max(now, next.start)));
        }
    }
    readonly Queue<Prepared> queue = new Queue<Prepared>();
    readonly float[] applied = new float[14];
    readonly float[] neutral = new float[14];
    AudioSource[] sources;
    CS_CommU robot;
    bool previousLookAt, ownsTarget, responseActive, generationComplete, hasStarted, waitingForRefill;
    string activeTurn;
    double queuedSeconds, firstArrival, lastArrival, refillStarted;
    class Prepared
    {
        public ConversationMessage message;
        public AudioClip clip;
        public double Duration => (double)message.sample_count / message.sample_rate;
    }
    class Scheduled
    {
        public Prepared prepared;
        public int source;
        public double start;
        public double End => start + prepared.Duration;
    }
    Scheduled current, next;

    void Start() { Acquire(); }
    public void Acquire()
    {
        if (ownsTarget) return;
        if (target == null) target = FindAnyObjectByType<CS_TestRotateBones>();
        if (target == null || target.Bone == null || target.Bone.Length < 10)
            throw new InvalidOperationException("Assign the CommU controller with ten bone references");
        for (int i = 1; i < 10; i++) if (target.Bone[i] == null) throw new InvalidOperationException("Missing CommU bone reference");
        if (!target.BeginExternalControl()) throw new InvalidOperationException("Another controller already owns CommU");
        ownsTarget = true;
        Array.Copy(target.JointAngles, applied, 14);
        Array.Copy(target.JointAngles, neutral, 14);
        robot = target.GetComponentInParent<CS_CommU>();
        if (robot != null) { previousLookAt = robot.enableLookAt; robot.enableLookAt = false; }
        if (sources == null) sources = new AudioSource[2];
        for (int i = 0; i < 2; i++)
        {
            if (sources[i] == null) sources[i] = gameObject.AddComponent<AudioSource>();
            sources[i].playOnAwake = false; sources[i].spatialBlend = 0; sources[i].pitch = 1;
        }
    }

    void Timing(string detail)
    { if (logTiming) Debug.Log($"[CommU playback] turn={activeTurn} {detail}"); }

    public void Enqueue(ConversationMessage message)
    {
        if (!ConversationProtocol.ValidateChunk(message)) throw new FormatException("Invalid audio/motion packet");
        if (!ownsTarget) Acquire();
        if (queue.Count >= 32) throw new InvalidOperationException("Speech queue is full");
        if (responseActive && (message.turn_id != activeTurn || generationComplete))
            throw new InvalidOperationException("Unexpected speech turn or audio after completion");
        // Decode and create clips on arrival, before their scheduling deadline.
        float[] audio = ConversationProtocol.DecodeAudio(message);
        var clip = AudioClip.Create("CommU " + message.turn_id, audio.Length, 1, message.sample_rate, false);
        clip.SetData(audio, 0);
        double now = AudioSettings.dspTime;
        if (!responseActive)
        {
            responseActive = true; activeTurn = message.turn_id;
            generationComplete = hasStarted = waitingForRefill = false;
            UnderrunCount = 0; firstArrival = lastArrival = now;
        }
        var prepared = new Prepared { message = message, clip = clip };
        queue.Enqueue(prepared); queuedSeconds += prepared.Duration;
        Timing($"event=chunk seq={message.sequence} arrival_gap_ms={(now - lastArrival) * 1000:F1} " +
            $"buffer_s={BufferedSeconds:F3} backend_ms={message.backend_elapsed_ms:F1} ready_ms={message.gemini_ready_ms:F1} " +
            $"gemini_gap_ms={message.gemini_gap_ms:F1} queue_ms={message.motion_queue_ms:F1} motion_ms={message.motion_inference_ms:F1}");
        lastArrival = now;
    }

    public void CompleteResponse()
    {
        generationComplete = true;
        Timing($"event=generation_complete buffer_s={BufferedSeconds:F3}");
    }

    Prepared Take()
    {
        var prepared = queue.Dequeue(); queuedSeconds -= prepared.Duration;
        return prepared;
    }

    Scheduled Schedule(Prepared prepared, int source, double start)
    {
        sources[source].clip = prepared.clip;
        sources[source].PlayScheduled(start);
        Timing($"event=schedule seq={prepared.message.sequence} lead_ms={(start - AudioSettings.dspTime) * 1000:F1} buffer_s={BufferedSeconds + prepared.Duration:F3}");
        return new Scheduled { prepared = prepared, start = start, source = source };
    }

    void LateUpdate()
    {
        if (!ownsTarget) return;
        double now = AudioSettings.dspTime;
        // Retire consumed clips before reusing their AudioSources, including after a slow frame.
        while (current != null && now >= current.End)
        {
            ConversationProtocol.Interpolate(current.prepared.message, current.prepared.Duration, applied);
            Release(current); current = next; next = null;
        }
        if (current == null && responseActive)
        {
            bool finished = generationComplete && queue.Count == 0;
            if (finished)
            {
                Timing($"event=playback_complete underruns={UnderrunCount}");
                responseActive = false; waitingForRefill = false;
            }
            else
            {
                if (hasStarted && !waitingForRefill)
                {
                    waitingForRefill = true; refillStarted = now; UnderrunCount++;
                    Timing($"event=underrun count={UnderrunCount} buffer_s={BufferedSeconds:F3}");
                }
                double required = hasStarted ? rebufferSeconds : startupBufferSeconds;
                // Short/final responses flush as soon as all audio and motion are known.
                if (queue.Count > 0 && (generationComplete || queuedSeconds + 1e-6 >= required))
                {
                    Timing($"event={(hasStarted ? "resume" : "start")} wait_ms={(now - (hasStarted ? refillStarted : firstArrival)) * 1000:F1} buffer_s={BufferedSeconds:F3}");
                    current = Schedule(Take(), 0, now + Math.Max(.01, prebufferSeconds));
                    hasStarted = true; waitingForRefill = false;
                }
            }
        }
        if (current != null && next == null && queue.Count > 0)
        {
            // Only schedule a contiguous continuation if its deadline can still be met.
            // Otherwise finish this clip and refill the reserve before resuming.
            if (current.End - AudioSettings.dspTime >= Math.Max(.01, prebufferSeconds))
                next = Schedule(Take(), 1 - current.source, current.End);
        }
        if (current != null && now >= current.start)
        {
            var source = sources[current.source];
            double seconds = source.isPlaying ? (double)source.timeSamples / current.prepared.message.sample_rate : now - current.start;
            ConversationProtocol.Interpolate(current.prepared.message, seconds, applied);
        }
        else if (!responseActive && current == null)
            for (int j = 0; j < 14; j++) applied[j] = Mathf.MoveTowards(applied[j], neutral[j], returnSpeed * Time.unscaledDeltaTime);
        // While waiting for more speech, preserve the last spoken pose, including at underruns.
        target.ApplyExternalPose(applied);
    }

    void DestroyClip(AudioClip clip)
    { if (Application.isPlaying) Destroy(clip); else DestroyImmediate(clip); }

    void Release(Scheduled value)
    {
        if (value == null) return;
        sources[value.source].Stop(); sources[value.source].clip = null;
        DestroyClip(value.prepared.clip);
    }

    public void StopAll()
    {
        while (queue.Count > 0) DestroyClip(queue.Dequeue().clip);
        queuedSeconds = 0;
        Release(current); Release(next); current = next = null;
        responseActive = generationComplete = hasStarted = waitingForRefill = false;
        activeTurn = null; UnderrunCount = 0;
    }

    public void ReleaseControl()
    {
        StopAll();
        if (ownsTarget && target != null) target.EndExternalControl();
        if (robot != null) robot.enableLookAt = previousLookAt;
        ownsTarget = false;
    }
    void OnDisable() { ReleaseControl(); }
}
