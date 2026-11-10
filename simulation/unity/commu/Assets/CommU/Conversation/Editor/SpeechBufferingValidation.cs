using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Deliberately delay packets in muted Play Mode. No network or paid API is needed.
[InitializeOnLoad]
public static class SpeechBufferingValidation
{
    const string Key = "CommUSpeechBufferingCheck";
    static SpeechBufferingValidation()
    { if (SessionState.GetBool(Key, false)) EditorApplication.update += Tick; }

    public static void RunBatch()
    {
        ConversationSceneSetup.Open();
        var client = UnityEngine.Object.FindAnyObjectByType<ConversationClient>();
        client.enabled = false; client.microphone.enabled = false;
        var ui = UnityEngine.Object.FindAnyObjectByType<ConversationUI>();
        if (ui != null) ui.enabled = false;
        client.player.startupBufferSeconds = .3f;
        client.player.rebufferSeconds = .3f;
        client.player.prebufferSeconds = .05;
        client.player.returnSpeed = 1000;
        SessionState.SetBool(Key, true); SessionState.SetInt(Key + "Phase", 0);
        SessionState.SetFloat(Key + "Deadline", (float)EditorApplication.timeSinceStartup + 60);
        EditorApplication.update -= Tick; EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    static ConversationMessage Chunk(string turn, int sequence, double duration = .15)
    {
        var audio = new float[(int)Math.Round(duration * 24000)];
        for (int i = 0; i < audio.Length; i++) audio[i] = .05f * Mathf.Sin(i * 2 * Mathf.PI * 440 / 24000);
        var angles = new float[28]; angles[7] = angles[21] = 6;
        return new ConversationMessage { type = "response_chunk", turn_id = turn, sequence = sequence,
            sample_rate = 24000, sample_count = audio.Length, audio_pcm16 = ConversationProtocol.EncodeAudio(audio),
            joint_names = ConversationProtocol.JointNames, timestamps = new double[] { 0, duration / 2 }, angles = angles };
    }

    static bool AudioPlaying(SpeechMotionPlayer player) => !player.IsBuffering && player.GetComponents<AudioSource>().Any(x => x.isPlaying);
    static void Check(bool value, string reason)
    { if (!value) throw new Exception("Speech buffering validation: " + reason); }
    static void Phase(int phase)
    {
        SessionState.SetInt(Key + "Phase", phase);
        SessionState.SetFloat(Key + "Since", (float)EditorApplication.timeSinceStartup);
    }

    static void Tick()
    {
        try
        {
            Check(EditorApplication.timeSinceStartup < SessionState.GetFloat(Key + "Deadline", 0), "Timed out");
            if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
            AudioListener.volume = 0;
            var player = UnityEngine.Object.FindAnyObjectByType<SpeechMotionPlayer>();
            if (player == null || !player.OwnsController) return;
            int phase = SessionState.GetInt(Key + "Phase", 0);
            double waited = EditorApplication.timeSinceStartup - SessionState.GetFloat(Key + "Since", 0);
            if (phase == 0)
            { player.Enqueue(Chunk("delayed", 0)); Phase(1); }
            else if (phase == 1 && waited > .25)
            {
                Check(player.IsBusy && player.IsBuffering && !AudioPlaying(player), "First packet must wait for reserve");
                player.Enqueue(Chunk("delayed", 1)); Phase(2);
            }
            else if (phase == 2 && AudioPlaying(player)) Phase(3);
            else if (phase == 3 && player.IsRebuffering)
            {
                Check(player.UnderrunCount == 1 && player.IsBusy, "Underrun must keep the response active");
                Check(Mathf.Abs(player.target.JointAngles[7] - 6) < .01, "Final spoken pose must be latched");
                player.Enqueue(Chunk("delayed", 2)); Phase(4);
            }
            else if (phase == 4 && waited > .25)
            {
                Check(player.IsRebuffering && !AudioPlaying(player), "One late packet must wait for refill");
                Check(player.UnderrunCount == 1, "Only count an underrun once");
                Check(Mathf.Abs(player.target.JointAngles[7] - 6) < .01, "Hold pose while refilling");
                player.Enqueue(Chunk("delayed", 3)); Phase(5);
            }
            else if (phase == 5 && AudioPlaying(player))
            { player.CompleteResponse(); Phase(6); }
            else if (phase == 6 && !player.IsBusy)
            {
                Check(player.UnderrunCount == 1, "Prepared adjacent chunks must play without another underrun");
                player.Enqueue(Chunk("short", 0, .2)); player.CompleteResponse(); Phase(7);
            }
            else if (phase == 7 && AudioPlaying(player)) Phase(8);
            else if (phase == 8 && !player.IsBusy)
            {
                Check(player.UnderrunCount == 0, "Short complete reply must flush below startup threshold");
                player.Enqueue(Chunk("cancelled", 0)); player.StopAll();
                Check(!player.IsBusy && player.BufferedSeconds == 0 && !AudioPlaying(player), "Cancel while buffered");
                player.Enqueue(Chunk("new", 0)); Phase(9);
            }
            else if (phase == 9 && waited > .25)
            {
                Check(player.IsBuffering && !AudioPlaying(player) && player.UnderrunCount == 0, "New turn must reset completion state");
                player.Enqueue(Chunk("new", 1)); Phase(10);
            }
            else if (phase == 10 && AudioPlaying(player))
            {
                player.StopAll();
                Check(!player.IsBusy && !AudioPlaying(player), "Cancel during playback");
                player.Enqueue(Chunk("late-complete", 0)); player.Enqueue(Chunk("late-complete", 1)); Phase(11);
            }
            else if (phase == 11 && player.IsRebuffering)
            {
                Check(player.IsBusy, "An empty unfinished stream must not report playback done");
                player.CompleteResponse(); Phase(12);
            }
            else if (phase == 12 && !player.IsBusy)
            {
                Debug.Log("Speech buffering Play Mode PASSED: startup reserve, contiguous segments, underrun, pose hold, refill, short replies and cancellation.");
                Finish(0);
            }
        }
        catch (Exception error) { Debug.LogException(error); Finish(1); }
    }

    static void Finish(int code)
    {
        SessionState.EraseBool(Key); EditorApplication.update -= Tick;
        AudioListener.volume = 1; EditorApplication.Exit(code);
    }
}
