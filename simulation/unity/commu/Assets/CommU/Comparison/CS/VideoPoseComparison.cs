using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Video;

[DefaultExecutionOrder(180)]
public class VideoPoseComparison : MonoBehaviour
{
    public string runFolder = "dataset/lucy/generation_runs/speech2gesture/00271_6f35a901156dd4cd1fdf";
    public CS_TestRotateBones target;
    public Camera robotCamera;
    public bool loop = true;
    public bool muted;
    public bool Ready { get; private set; }
    public bool Seeking { get; private set; }
    public string Status { get; private set; } = "Preparing videos";
    public double Duration { get; private set; }
    public double Position => generated != null && generated.isPrepared ? generated.time : 0;
    public bool IsPlaying => generated != null && generated.isPlaying;
    public VideoPlayer OriginalPlayer => original;
    public VideoPlayer GeneratedPlayer => generated;
    public ComparisonRun Run { get; private set; }
    readonly float[] pose = new float[14];
    VideoPlayer original, generated;
    RenderTexture originalTexture, generatedTexture, robotTexture;
    RenderTexture previousTarget;
    CS_CommU robot;
    bool ownsTarget, previousLookAt, originalSeeked, generatedSeeked, endReached;

    IEnumerator Start()
    {
        try
        {
            Run = new ComparisonRun(runFolder);
            if (target == null || robotCamera == null) throw new InvalidOperationException("Assign robot controller and camera");
            if (!target.BeginExternalControl()) throw new InvalidOperationException("Robot is already controlled");
            ownsTarget = true;
            robot = target.GetComponentInParent<CS_CommU>();
            if (robot != null) { previousLookAt = robot.enableLookAt; robot.enableLookAt = false; }
            previousTarget = robotCamera.targetTexture;
            originalTexture = MakeTexture(); generatedTexture = MakeTexture(); robotTexture = MakeTexture();
            robotCamera.targetTexture = robotTexture;
            original = MakePlayer(Run.Original, originalTexture, false);
            generated = MakePlayer(Run.Generated, generatedTexture, true);
            original.seekCompleted += _ => originalSeeked = true;
            generated.seekCompleted += _ => generatedSeeked = true;
            generated.loopPointReached += _ => endReached = true;
            original.Prepare(); generated.Prepare();
        }
        catch (Exception error) { Error(error.Message); yield break; }
        double deadline = Time.realtimeSinceStartupAsDouble + 30;
        while (!original.isPrepared || !generated.isPrepared)
        {
            if (Status.StartsWith("Error:")) yield break;
            if (Time.realtimeSinceStartupAsDouble > deadline) { Error("Video preparation timed out"); yield break; }
            yield return null;
        }
        Duration = Math.Min(Run.Poses.Duration, Math.Min(generated.length, original.length - Run.OriginalOffset));
        if (Duration <= 0) { Error("Videos have no shared timeline"); yield break; }
        // The generated video is the master clock. Original follows the manifest's source offset.
        original.timeReference = VideoTimeReference.ExternalTime;
        original.externalReferenceTime = Run.OriginalOffset;
        Ready = true;
        yield return SeekRoutine(0, false);
        Debug.Log($"Comparison ready: {Run.Poses.Count} poses, {Duration:F3}s, original offset {Run.OriginalOffset:F3}s");
    }

    RenderTexture MakeTexture()
    {
        var texture = new RenderTexture(1280, 720, 24, RenderTextureFormat.ARGB32);
        texture.Create();
        return texture;
    }

    VideoPlayer MakePlayer(string path, RenderTexture texture, bool audio)
    {
        var player = gameObject.AddComponent<VideoPlayer>();
        player.playOnAwake = false; player.isLooping = false; player.waitForFirstFrame = true;
        player.source = VideoSource.Url; player.url = new Uri(path).AbsoluteUri;
        player.renderMode = VideoRenderMode.RenderTexture; player.targetTexture = texture;
        player.audioOutputMode = audio ? VideoAudioOutputMode.Direct : VideoAudioOutputMode.None;
        if (audio) { player.controlledAudioTrackCount = 1; player.SetDirectAudioMute(0, muted); }
        player.errorReceived += (_, message) => Error(message);
        return player;
    }

    void Error(string message)
    {
        Status = "Error: " + message; Ready = false;
        original?.Pause(); generated?.Pause();
        Debug.LogError(Status);
    }

    void LateUpdate()
    {
        if (!Ready || Seeking) return;
        double seconds = Math.Max(0, Math.Min(Position, Duration));
        original.externalReferenceTime = seconds + Run.OriginalOffset;
        Run.Poses.Sample(seconds, pose);
        target.ApplyExternalPose(pose);
        if ((IsPlaying && seconds >= Duration - .001) || endReached)
        {
            endReached = false;
            Pause();
            if (loop) Seek(0, true);
            else Status = "Completed";
        }
    }

    public void Play()
    {
        if (!Ready || Seeking) return;
        if (Position >= Duration - 1.0 / 30) { Seek(0, true); return; }
        original.externalReferenceTime = Position + Run.OriginalOffset;
        original.Play(); generated.Play(); Status = "Playing";
    }
    public void Pause()
    { original?.Pause(); generated?.Pause(); if (Ready) Status = "Paused"; }
    public void Seek(double seconds, bool resume)
    { if (Ready && !Seeking) StartCoroutine(SeekRoutine(seconds, resume)); }

    IEnumerator SeekRoutine(double seconds, bool resume)
    {
        Seeking = true; Pause(); Status = "Seeking";
        seconds = Math.Max(0, Math.Min(seconds, Math.Max(0, Duration - 1.0 / 30)));
        originalSeeked = Math.Abs(original.time - seconds - Run.OriginalOffset) < .001;
        generatedSeeked = Math.Abs(generated.time - seconds) < .001;
        original.externalReferenceTime = seconds + Run.OriginalOffset;
        if (!originalSeeked) original.time = seconds + Run.OriginalOffset;
        if (!generatedSeeked) generated.time = seconds;
        // On macOS seekCompleted can precede presentation while paused. Decode silently
        // until both players actually present the requested time, then pause together.
        generated.SetDirectAudioMute(0, true);
        original.Play(); generated.Play();
        double deadline = Time.realtimeSinceStartupAsDouble + 10;
        while (!originalSeeked || !generatedSeeked || original.frame < 0 || generated.frame < 0 ||
               Math.Abs(generated.time - seconds) > .07 || Math.Abs(original.time - seconds - Run.OriginalOffset) > .1)
        {
            if (Time.realtimeSinceStartupAsDouble > deadline) { Seeking = false; Error("Video seek timed out"); yield break; }
            yield return null;
        }
        original.Pause(); generated.Pause();
        generated.SetDirectAudioMute(0, muted);
        Run.Poses.Sample(Position, pose); target.ApplyExternalPose(pose);
        Debug.Log($"Comparison seek completed: requested={seconds:F3}, generated={generated.time:F3}, original={original.time:F3}, frames={generated.frame}/{original.frame}");
        endReached = false; Seeking = false; Status = "Paused";
        if (resume) Play();
    }

    void OnGUI()
    {
        GUI.Box(new Rect(0, 0, Screen.width, Screen.height), "");
        GUI.Label(new Rect(15, 8, Screen.width - 30, 28), "Original → Lucy 2.6 → DINOv3 pose estimation → Unity CommU");
        float width = (Screen.width - 40) / 3f;
        float height = Mathf.Max(50, Screen.height - 145);
        Panel(new Rect(10, 40, width, height), "Original video", originalTexture);
        Panel(new Rect(20 + width, 40, width, height), "Generated video", generatedTexture);
        Panel(new Rect(30 + 2 * width, 40, width, height), "Unity robot · predicted 14 joint angles", robotTexture);
        GUILayout.BeginArea(new Rect(15, Screen.height - 95, Screen.width - 30, 90));
        GUILayout.Label($"{Status}   {Position:F2} / {Duration:F2} s");
        GUILayout.BeginHorizontal();
        GUI.enabled = Ready && !Seeking;
        if (GUILayout.Button(IsPlaying ? "Pause" : "Play", GUILayout.Width(90))) { if (IsPlaying) Pause(); else Play(); }
        if (GUILayout.Button("Restart", GUILayout.Width(90))) Seek(0, true);
        loop = GUILayout.Toggle(loop, "Loop", GUILayout.Width(65));
        bool nextMute = GUILayout.Toggle(muted, "Mute", GUILayout.Width(65));
        if (nextMute != muted) { muted = nextMute; generated?.SetDirectAudioMute(0, muted); }
        float nextTime = GUILayout.HorizontalSlider((float)Position, 0, (float)Math.Max(.01, Duration));
        if (Ready && Math.Abs(nextTime - Position) > .08) Seek(nextTime, IsPlaying);
        GUI.enabled = true;
        GUILayout.EndHorizontal();
        GUILayout.Label("Audio: generated video only. Motion: existing pose-estimation CSV (degrees), clamped to Unity joint limits.");
        GUILayout.EndArea();
    }

    void Panel(Rect area, string title, Texture texture)
    {
        GUI.Box(area, "");
        GUI.Label(new Rect(area.x + 8, area.y + 5, area.width - 16, 24), title);
        if (texture != null) GUI.DrawTexture(new Rect(area.x + 4, area.y + 32, area.width - 8, area.height - 36), texture, ScaleMode.ScaleToFit);
    }

    void OnDisable()
    {
        Ready = false;
        StopAllCoroutines(); original?.Stop(); generated?.Stop();
        if (ownsTarget && target != null) target.EndExternalControl();
        ownsTarget = false;
        if (robot != null) robot.enableLookAt = previousLookAt;
        if (robotCamera != null) robotCamera.targetTexture = previousTarget;
        foreach (var texture in new[] { originalTexture, generatedTexture, robotTexture })
            if (texture != null) { texture.Release(); Destroy(texture); }
    }
}
