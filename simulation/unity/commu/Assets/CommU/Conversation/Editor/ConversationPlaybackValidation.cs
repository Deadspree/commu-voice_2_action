using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Optional integration check against a running --demo-audio backend on port 8765.
[InitializeOnLoad]
public static class ConversationPlaybackValidation
{
    const string Key = "CommUConversationPlaybackCheck";
    static ConversationPlaybackValidation()
    {
        if (SessionState.GetBool(Key, false)) EditorApplication.update += Tick;
    }

    public static void RunBatch()
    {
        ConversationSceneSetup.Open();
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, "--conversation-backend");
        if (index >= 0 && index + 1 < args.Length)
            UnityEngine.Object.FindAnyObjectByType<ConversationClient>().backendUrl = args[index + 1];
        SessionState.SetBool(Key, true);
        SessionState.SetInt(Key + "Phase", 0);
        SessionState.SetBool(Key + "Motion", false);
        SessionState.SetBool(Key + "Audio", false);
        SessionState.SetFloat(Key + "Deadline", (float)EditorApplication.timeSinceStartup + 45);
        EditorApplication.update -= Tick;
        EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    static void Tick()
    {
        try
        {
            if (EditorApplication.timeSinceStartup > SessionState.GetFloat(Key + "Deadline", 0))
                throw new Exception("Timed out waiting for demo speech playback; check backend and audio device");
            if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
            AudioListener.volume = 0; // Exercise the clock without playing sound on the user's speakers.
            var client = UnityEngine.Object.FindAnyObjectByType<ConversationClient>();
            if (client == null) return;
            if (client.state.status.StartsWith("Error:")) throw new Exception(client.state.status);
            int phase = SessionState.GetInt(Key + "Phase", 0);
            if (phase == 0 && client.state.connected)
            {
                if (!client.state.demo) throw new Exception("Run backend with --demo-audio for this check");
                SessionState.SetString(Key + "Initial", JsonUtility.ToJson(new ConversationMessage { angles = (float[])client.player.target.JointAngles.Clone() }));
                client.Preview(); SessionState.SetInt(Key + "Phase", 1);
            }
            else if (phase == 1 && client.player.IsBusy) SessionState.SetInt(Key + "Phase", 2);
            else if (phase == 2)
            {
                float[] initial = JsonUtility.FromJson<ConversationMessage>(SessionState.GetString(Key + "Initial", "")).angles;
                if (initial.Where((value, i) => Mathf.Abs(value - client.player.target.JointAngles[i]) > .01f).Any())
                    SessionState.SetBool(Key + "Motion", true);
                if (client.GetComponents<AudioSource>().Any(source => source.isPlaying)) SessionState.SetBool(Key + "Audio", true);
                if (!client.player.IsBusy && client.state.status == "Listening")
                {
                    if (!SessionState.GetBool(Key + "Motion", false)) throw new Exception("No model-driven joint movement observed");
                    if (!SessionState.GetBool(Key + "Audio", false)) throw new Exception("AudioSource never entered playback");
                    client.Preview(); SessionState.SetInt(Key + "Phase", 3);
                }
            }
            else if (phase == 3 && client.player.IsBusy)
            {
                client.CancelTurn();
                if (client.player.IsBusy || client.GetComponents<AudioSource>().Any(source => source.isPlaying))
                    throw new Exception("Cancellation did not stop audio and motion together");
                Debug.Log("CommU live conversation playback PASSED: WebSocket, trained-model motion, audio clock, completion and interruption.");
                Finish(0);
            }
        }
        catch (Exception error) { Debug.LogException(error); Finish(1); }
    }

    static void Finish(int code)
    {
        SessionState.EraseBool(Key);
        EditorApplication.update -= Tick;
        AudioListener.volume = 1;
        EditorApplication.Exit(code);
    }
}
