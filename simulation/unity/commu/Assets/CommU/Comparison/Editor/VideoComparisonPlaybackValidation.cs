using System;
using System.IO;
using UnityEditor;
using UnityEngine;

[InitializeOnLoad]
public static class VideoComparisonPlaybackValidation
{
    const string Key = "CommUVideoComparisonCheck";
    static VideoComparisonPlaybackValidation()
    { if (SessionState.GetBool(Key, false)) EditorApplication.update += Tick; }

    public static void RunBatch()
    {
        VideoComparisonSetup.Open();
        var view = UnityEngine.Object.FindAnyObjectByType<VideoPoseComparison>();
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, "--comparison-run");
        if (index >= 0) view.runFolder = args[index + 1];
        view.muted = true; view.loop = false;
        SessionState.SetBool(Key, true);
        SessionState.SetInt(Key + "Phase", 0);
        SessionState.SetFloat(Key + "Deadline", (float)EditorApplication.timeSinceStartup + 90);
        EditorApplication.update -= Tick; EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    static void Check(bool value, string reason)
    { if (!value) throw new Exception("Video comparison playback: " + reason); }

    static void CheckPose(VideoPoseComparison view)
    {
        var expected = new float[14];
        view.Run.Poses.Sample(view.Position, expected);
        for (int j = 0; j < 14; j++)
            Check(Mathf.Abs(view.target.JointAngles[j] - CS_JointLimitDefinition.ClampGoal(j, expected[j])) < .05f,
                "Robot pose differs from the generated video's timestamp at joint " + j);
    }

    static void Tick()
    {
        try
        {
            Check(EditorApplication.timeSinceStartup < SessionState.GetFloat(Key + "Deadline", 0), "Timed out");
            if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
            var view = UnityEngine.Object.FindAnyObjectByType<VideoPoseComparison>();
            if (view == null) return;
            Check(!view.Status.StartsWith("Error:"), view.Status);
            int phase = SessionState.GetInt(Key + "Phase", 0);
            if (phase == 0 && view.Ready && !view.Seeking)
            { view.Play(); SessionState.SetInt(Key + "Phase", 1); }
            else if (phase == 1 && view.Position > 1)
            {
                Check(view.OriginalPlayer.frame > 0 && view.GeneratedPlayer.frame > 0, "Both videos must decode");
                Check(Math.Abs(view.OriginalPlayer.time - view.Position - view.Run.OriginalOffset) < .15, "Video synchronization");
                view.Pause();
                SessionState.SetFloat(Key + "Position", (float)view.Position);
                SessionState.SetFloat(Key + "PausedAt", (float)EditorApplication.timeSinceStartup);
                SessionState.SetInt(Key + "Phase", 2);
            }
            else if (phase == 2 && EditorApplication.timeSinceStartup - SessionState.GetFloat(Key + "PausedAt", 0) > .3)
            {
                Check(Math.Abs(view.Position - SessionState.GetFloat(Key + "Position", 0)) < .04, "Pause must hold time");
                CheckPose(view);
                view.Seek(view.Duration / 2, false);
                SessionState.SetInt(Key + "Phase", 3);
            }
            else if (phase == 3 && !view.Seeking)
            {
                Check(Math.Abs(view.Position - view.Duration / 2) < 2.5 / 30, $"Seek position: actual={view.Position:F3}, expected={view.Duration / 2:F3}");
                Check(Math.Abs(view.OriginalPlayer.time - view.Position - view.Run.OriginalOffset) < .1, "Videos must agree after seeking");
                CheckPose(view); view.Play(); SessionState.SetInt(Key + "Phase", 4);
            }
            else if (phase == 4 && view.Position > view.Duration / 2 + .4)
            {
                Check(view.IsPlaying, "Resume after seek");
                Debug.Log("Video comparison Play Mode PASSED: both videos, shared timing, joint replay, pause, seek and resume.");
                Finish(0);
            }
        }
        catch (Exception error) { Debug.LogException(error); Finish(1); }
    }

    static void Finish(int code)
    { SessionState.EraseBool(Key); EditorApplication.update -= Tick; EditorApplication.Exit(code); }
}
