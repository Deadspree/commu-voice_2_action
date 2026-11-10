using System;
using UnityEngine;

public class ConversationState : MonoBehaviour
{
    public string status = "Disconnected";
    [TextArea] public string userTranscript = "";
    [TextArea] public string robotTranscript = "";
    public bool connected;
    public bool demo;
    public string activeTurn;
    public string NewTurn() { activeTurn = Guid.NewGuid().ToString("N"); return activeTurn; }
}
