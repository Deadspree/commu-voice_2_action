using UnityEngine;
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
using UnityEngine.InputSystem;
#endif

public class ConversationUI : MonoBehaviour
{
    public ConversationClient client;
    string input = "Hello CommU! How are you?";
    void Update()
    {
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
        bool down = Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame;
        bool up = Keyboard.current != null && Keyboard.current.spaceKey.wasReleasedThisFrame;
#else
        bool down = Input.GetKeyDown(KeyCode.Space);
        bool up = Input.GetKeyUp(KeyCode.Space);
#endif
        if (GUIUtility.keyboardControl != 0) return;
        if (down) client.BeginRecording();
        if (up) client.microphone.Finish();
    }
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(15, 15, 420, 350), GUI.skin.box);
        GUILayout.Label("CommU • Gemini Live 3.8");
        GUILayout.Label("AI-generated voice • " + client.state.status);
        if (GUILayout.Button(client.state.connected ? "Reconnect" : "Connect")) client.Connect();
        GUI.enabled = client.state.connected;
        if (client.state.demo)
        { if (GUILayout.Button("Preview audio + predicted motion")) client.Preview(); }
        else
        {
            GUILayout.Label("Hold Space to talk, or use the recording button.");
            if (GUILayout.Button(client.microphone.IsRecording ? "Finish speaking" : "Start speaking"))
            { if (client.microphone.IsRecording) client.microphone.Finish(); else client.BeginRecording(); }
            input = GUILayout.TextField(input, 2000);
            if (GUILayout.Button("Send text")) client.SendText(input);
        }
        if (GUILayout.Button("Stop / interrupt")) client.CancelTurn();
        GUI.enabled = true;
        GUILayout.Label("You: " + client.state.userTranscript);
        GUILayout.Label("CommU: " + client.state.robotTranscript);
        GUILayout.EndArea();
    }
}
